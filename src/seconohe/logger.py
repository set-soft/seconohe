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
from .comfy_notification import send_toast_notification


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
        colorama_init()
    except ImportError:
        # If the import fails, import our fallback module
        from . import ansi
        # Assign the names from our fallback module's namespace
        Fore = ansi.Fore
        Back = ansi.Back
        Style = ansi.Style

        # Define a dummy init function to ensure it always exists
        def colorama_init() -> None:
            pass


class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors"""

    def __init__(self, name: str) -> None:
        super(logging.Formatter, self).__init__()
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
        self.standalone = True

    def format(self, record: logging.LogRecord) -> str:
        formats = self.FORMATS_STANDALONE if self.standalone else self.FORMATS
        log_fmt = formats.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def on_log_error_or_warning(logger: logging.Logger, record: logging.LogRecord) -> None:
    """
    This function is called whenever a log with level WARNING or higher is emitted.
    The 'record' object contains all information about the log event.
    """
    if record.levelno == logging.WARNING:
        summary = "Warning"
        severity = "warn"
    else:
        summary = "Error"
        severity = "error"
    send_toast_notification(logger, record.getMessage(), summary=summary, severity=severity)


class WarningAndErrorFilter(logging.Filter):
    """
    A custom log filter that intercepts logs of a certain level.
    """
    def __init__(self, logger: logging.Logger, callback: Callable, level: int = logging.WARNING):
        """
        Initializes the filter.

        Args:
            callback: The function to call when a log record meets the level criteria.
            level: The minimum level to trigger the callback.
        """
        super().__init__()
        self._callback = callback
        self._level = level
        self._logger = logger

    def filter(self, record: logging.LogRecord) -> bool:
        """
        This method is called for every log record.
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
    NODES_DEBUG_VAR = name.upper() + "_NODES_DEBUG"
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
    """ Change the logger to standalone CLI mode.
        args.verbose is the verbosity level
        args.quiet is the optional quiet mode (warning level) """
    if hasattr(logger, 'custom_formatter'):
        logger.custom_formatter.set_standalone()
    if hasattr(args, 'quiet'):
        logger.setLevel(logging.WARNING)
    else:
        verbose = args.verbose
        logger.setLevel(logging.DEBUG - (verbose - 1) if verbose else logging.INFO)


def get_debug_level(logger: logging.Logger) -> int:
    return logging.DEBUG - logger.getEffectiveLevel() + 1


def debugl(logger: logging.Logger, level: int, msg: str) -> None:
    if get_debug_level(logger) >= level:
        logger.debug(msg)
