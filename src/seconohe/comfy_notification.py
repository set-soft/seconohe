# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
# License: GPLv3
# Project: SeCoNoHe
#
# ComfyUI Toast API messages
# Original code from Gemini 2.5 Pro, which was really outdated
# Took ideas from Easy Use nodes and looking at ComfyUI code
import logging
from typing import Optional
# ComfyUI imports
try:
    from server import PromptServer
    with_comfy = True
except Exception:
    with_comfy = False
# Local imports
from . import settings

logger = logging.getLogger(f"{settings.NODES_NAME}.comfy_notification")
EVENT_NAME = "seconohe-toast"


def send_toast_notification(message: str, summary: str = "Warning", severity: str = "warn", sid: Optional[str] = None):
    """
    Sends a toast notification event to the ComfyUI client.

    Args:
        message (str): The message content of the toast.
        severity (str): The type of toast. Can be 'success' | 'info' | 'warn' | 'error' | 'secondary' | 'contrast'
        summary (str): Short explanation
        sid (str, optional): The session ID of the client to send to.
                            If None, broadcasts to all clients. Defaults to None.
    """
    if not with_comfy:
        return
    try:
        PromptServer.instance.send_sync(
            EVENT_NAME,  # This is our custom event name
            {
                'message': message,
                'summary': summary,
                'severity': severity
            },
            sid
        )
    except Exception as e:
        logger.error(f"when trying to use ComfyUI PromptServer: {e}")
