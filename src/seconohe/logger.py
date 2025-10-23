# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnologïa Industrial
# License: GPL-3.0
# Project: SeCoNoHe
from __future__ import annotations  # Good practice
import argparse  # For typing
import logging
import os
import sys
from typing import TYPE_CHECKING, Callable
from .comfy_notification import send_toast_notification, ToastSeverity


# 1. This block is ONLY for the type checker.
#    It imports the canonical types and assigns them.
if TYPE_CHECKING:
    # Import the real objects for type analysis
    from colorama import Fore, Back, Style
    # A dummy init function hint
    def colorama_init() -> None: ...

# 2. This block is for RUNTIME.
#    It does not use `from ... import ...` for the conflicting names.
else:
    try:
        # Import the entire module
        import colorama
        # Assign the names from the module's namespace
        Fore = colorama.Fore
        Back = colorama.Back
        Style = colorama.Style
        colorama_init = colorama.init
        # Initialize at runtime
        # TODO: What about Windows? Should we filter the sequences?
        colorama_init(strip=False)
    except ImportError:
        # If the import fails, import our fallback module
        from . import _ansi
        # Assign the names from our fallback module's namespace
        Fore = _ansi.Fore
        Back = _ansi.Back
        Style = _ansi.Style

        # Define a dummy init function to ensure it always exists
        def colorama_init() -> None:
            pass


class CustomFormatter(logging.Formatter):
    """
    A logging Formatter to add colors and switch between ComfyUI and standalone formats.

    This formatter uses different log message formats and colors depending on whether
    it is running in a standard command-line interface (standalone) or as part of a
    ComfyUI node. It automatically handles color disabling for non-tty outputs.

    :ivar standalone: If ``True``, uses formats suitable for a standard CLI.
                      If ``False``, uses formats with a node name prefix for ComfyUI.
    :vartype standalone: bool
    """

    def __init__(self, name: str) -> None:
        """
        Initializes the formatter with two sets of format strings.

        :param name: The name of the node or tool, used as a prefix in log messages.
        :type name: str
        """
        super().__init__()
        white = Fore.WHITE + Style.BRIGHT
        yellow = Fore.YELLOW + Style.BRIGHT
        red = Fore.RED + Style.BRIGHT
        red_alarm = Fore.RED + Back.WHITE + Style.BRIGHT
        cyan = Fore.CYAN + Style.BRIGHT
        reset = Style.RESET_ALL
        # Node formats
        # format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s "
        #          "(%(filename)s:%(lineno)d)"
        format = f"[{name} %(levelname)s] %(message)s (%(name)s - %(filename)s:%(lineno)d)"
        format_simple = f"[{name}] %(message)s"
        self.FORMATS = {
            logging.DEBUG: cyan + format + reset,
            logging.INFO: white + format_simple + reset,
            logging.WARNING: yellow + format + reset,
            logging.ERROR: red + format + reset,
            logging.CRITICAL: red_alarm + format + reset
        }
        # Standalone formats (CLI)
        format = "[%(levelname)s] %(message)s (%(name)s - %(filename)s:%(lineno)d)"
        format_simple = "%(message)s"
        if not sys.stdout.isatty():
            # For standalone use we avoid colors on log files
            # On Comfy we never get a real TTY, Comfy gets the messages and then sends them to the console
            white = yellow = red = red_alarm = cyan = reset = ""
        self.FORMATS_STANDALONE = {
            logging.DEBUG: cyan + format + reset,
            logging.INFO: white + format_simple + reset,
            logging.WARNING: yellow + format + reset,
            logging.ERROR: red + format + reset,
            logging.CRITICAL: red_alarm + format + reset
        }
        # Assume we are a node
        self.standalone = False

    def set_standalone(self) -> None:
        """Switches the formatter to standalone (CLI) mode."""
        self.standalone = True

    def format(self, record: logging.LogRecord) -> str:
        formats = self.FORMATS_STANDALONE if self.standalone else self.FORMATS
        log_fmt = formats.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def on_log_error_or_warning(logger: logging.Logger, record: logging.LogRecord) -> None:
    """
    A callback function that sends a toast notification for WARNING or ERROR logs.

    This function is designed to be attached to a logger via the `WarningAndErrorFilter`.
    It translates log records into UI notifications for the ComfyUI frontend.

    :param logger: The logger instance that emitted the record.
    :type logger: logging.Logger
    :param record: The log record containing event information.
    :type record: logging.LogRecord
    """
    severity: ToastSeverity = "warn" if record.levelno == logging.WARNING else "error"
    summary = "Warning" if record.levelno == logging.WARNING else "Error"
    send_toast_notification(logger, record.getMessage(), summary=summary, severity=severity)


class WarningAndErrorFilter(logging.Filter):
    """
    A custom log filter that intercepts logs at a certain level and triggers a callback.
    """
    def __init__(self, logger: logging.Logger, callback: Callable[[logging.Logger, logging.LogRecord], None],
                 level: int = logging.WARNING):
        """
        Initializes the filter.

        :param logger: The logger instance that this filter will be associated with.
        :type logger: logging.Logger
        :param callback: The function to call when a log record meets the level criteria.
                         It should accept a logger and a LogRecord as arguments.
        :type callback: Callable[[logging.Logger, logging.LogRecord], None]
        :param level: The minimum logging level to trigger the callback.
                      Defaults to ``logging.WARNING``.
        :type level: int
        """
        super().__init__()
        self._callback = callback
        self._level = level
        self._logger = logger

    def filter(self, record: logging.LogRecord) -> bool:
        """
        This method is called for every log record passed to the logger.

        If the record's level is at or above the filter's threshold, it triggers
        the callback. It always returns ``True`` to allow the record to be
        processed by subsequent handlers.

        :param record: The log record to be processed.
        :type record: logging.LogRecord
        :return: ``True`` to allow the record to pass to the next handler.
        :rtype: bool
        """
        # Check if the log level is WARNING or higher
        if record.levelno >= self._level:
            self._callback(self._logger, record)

        # Always return True to ensure the log is always processed
        # by the handlers after this filter.
        return True


# ######################
# Logger setup
# ######################
def initialize_logger(name: str) -> logging.Logger:
    """
    Initializes and configures a logger for a SeCoNoHe-compatible node or tool.

    This function creates a logger with a custom color formatter and a filter
    that forwards warnings and errors as UI toast notifications. It also sets
    the logging level based on ComfyUI's global settings and a custom
    environment variable (`{NAME}_NODES_DEBUG`).

    :param name: The name for the logger, typically the name of the node suite.
    :type name: str
    :return: The configured logger instance.
    :rtype: logging.Logger
    """
    # Create a new logger
    logger = logging.getLogger(name)
    logger.propagate = False

    # Add the custom filter to the logger.
    logger.addFilter(WarningAndErrorFilter(logger, callback=on_log_error_or_warning))

    # Add handler if we don't have one.
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        custom_formatter = CustomFormatter(name)
        handler.setFormatter(custom_formatter)
        logger.addHandler(handler)
        setattr(logger, 'custom_formatter', custom_formatter)

    # Determine the ComfyUI global log level (influenced by --verbose)
    # comfy_root_logger = logging.getLogger('comfy')
    effective_comfy_level = logging.getLogger().getEffectiveLevel()

    # Check our custom environment variable for more verbosity
    NODES_DEBUG_VAR = name.replace('-', '_').upper() + "_NODES_DEBUG"
    try:
        nodes_debug_env = int(os.environ.get(NODES_DEBUG_VAR, "0"))
    except ValueError:
        nodes_debug_env = 0

    # Set node's logger level
    if nodes_debug_env:
        logger.setLevel(logging.DEBUG - (nodes_debug_env - 1))
        final_level_str = f"DEBUG (due to {NODES_DEBUG_VAR}={nodes_debug_env})"
    else:
        logger.setLevel(effective_comfy_level)
        final_level_str = logging.getLevelName(effective_comfy_level) + " (matching ComfyUI global)"

    # A temporary logger for this message
    _initial_setup_logger = logging.getLogger(name + ".setup")
    _initial_setup_logger.debug(f"{name} logger level set to: {final_level_str}")

    return logger


def logger_set_standalone(logger: logging.Logger, args: argparse.Namespace) -> None:
    """
    Switches a logger to standalone (CLI) mode and sets its verbosity.

    This should be called by command-line tools after parsing arguments. It
    configures the logger's formatter for standard console output and sets the
    logging level based on verbosity flags.

    :param logger: The logger instance to modify.
    :type logger: logging.Logger
    :param args: The parsed arguments object from `argparse`. Expected to have
                 a ``verbose`` attribute (int) and an optional ``quiet``
                 attribute (bool).
    :type args: argparse.Namespace
    """
    if hasattr(logger, 'custom_formatter'):
        logger.custom_formatter.set_standalone()
    if hasattr(args, 'quiet'):
        logger.setLevel(logging.WARNING)
    else:
        verbose = args.verbose
        logger.setLevel(logging.DEBUG - (verbose - 1) if verbose else logging.INFO)


def get_debug_level(logger: logging.Logger) -> int:
    """
    Calculates the custom verbosity level based on the logger's effective level.

    Assumes that higher verbosity corresponds to a lower logging level number
    (e.g., DEBUG=10, DEBUG-1=9 for level 2, etc.).

    :param logger: The logger instance to check.
    :type logger: logging.Logger
    :return: An integer representing the debug verbosity level (1, 2, ...).
    :rtype: int
    """
    return logging.DEBUG - logger.getEffectiveLevel() + 1


def debugl(logger: logging.Logger, level: int, msg: str, *args, **kwargs) -> None:
    """
    Logs a message with `logger.debug` but only if the logger's verbosity is
    at or above the specified level.

    :param logger: The logger instance to use.
    :type logger: logging.Logger
    :param level: The required verbosity level to display this message.
    :type level: int
    :param msg: The log message string.
    :type msg: str
    :param args: Positional arguments for the log message.
    :param kwargs: Keyword arguments for the log message.
    """
    if get_debug_level(logger) >= level:
        logger.debug(msg, *args, **kwargs)


__all__ = ['initialize_logger', 'logger_set_standalone', 'get_debug_level', 'debugl']
