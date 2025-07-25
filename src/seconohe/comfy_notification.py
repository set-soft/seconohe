# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
# License: GPLv3
# Project: SeCoNoHe
#
# ComfyUI Toast API messages
# Original code from Gemini 2.5 Pro, which was really outdated
# Took ideas from Easy Use nodes and looking at ComfyUI code
import logging
from typing import Optional, Literal
# ComfyUI imports
try:
    from server import PromptServer
    with_comfy = True
except Exception:
    with_comfy = False

_EVENT_NAME = "seconohe-toast"
ToastSeverity = Literal['success', 'info', 'warn', 'error', 'secondary', 'contrast']


def send_toast_notification(logger: logging.Logger, message: str, summary: str = "Warning", severity: ToastSeverity = "warn",
                            sid: Optional[str] = None) -> None:
    """
    Sends a toast notification event to the ComfyUI web client via WebSocket.

    This function communicates with the frontend JavaScript listener for the
    ``seconohe-toast`` event, triggering a UI notification. It will fail silently
    if not run within a ComfyUI server environment.

    :param logger: The logger instance to use for reporting errors if the send fails.
    :type logger: logging.Logger
    :param message: The main detail message to display in the toast.
    :type message: str
    :param summary: A short, bolded summary or title for the toast. Defaults to 'Warning'.
    :type summary: str
    :param severity: The style/color of the toast. Must be one of 'success',
                     'info', 'warn', 'error', 'secondary', or 'contrast'.
                     Defaults to 'warn'.
    :type severity: Literal['success', 'info', 'warn', 'error', 'secondary', 'contrast']
    :param sid: The session ID of a specific client to send the message to.
                If ``None``, the message is broadcast to all connected clients.
                Defaults to ``None``.
    :type sid: Optional[str]
    """
    # Check if we are in a ComfyUI environment and the server instance is available
    if not with_comfy or not hasattr(PromptServer, 'instance') or PromptServer.instance is None:
        logger.debug("Toast notification skipped: Not running in an active ComfyUI server environment.")
        return
    try:
        # Use the PromptServer instance to send a custom event to the frontend
        PromptServer.instance.send_sync(
            _EVENT_NAME,  # The custom event name the JS is listening for
            {
                'message': message,
                'summary': summary,
                'severity': severity
            },
            sid
        )
    except Exception as e:
        logger.error(f"Failed to send toast notification via ComfyUI PromptServer: {e}")
