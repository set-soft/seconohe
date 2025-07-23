# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
# License: GPLv3
# Project: ComfyUI-AudioSeparation
#
# ComfyUI Node actions
import logging
from typing import Optional
# ComfyUI imports
try:
    from server import PromptServer
    with_comfy = True
except Exception:
    with_comfy = False

EVENT_NAME = "seconohe-node"


def send_node_action(logger: logging.Logger, action: str, arg1: Optional[str] = None, arg2: Optional[str] = None,
                     sid: Optional[str] = None):
    """
    Sends a node action event to the ComfyUI client.

    Args:
        logger (logging.Logger): The logger used in case we need to report an error
        action (str): Action to be performed.
        arg1 (str): First argument
        arg2 (str): Second argument
        sid (str, optional): The session ID of the client to send to.
                            If None, broadcasts to all clients. Defaults to None.
    """
    # Check we have ComfyUI and the server is running
    if not with_comfy or not hasattr(PromptServer, 'instance'):
        return
    try:
        PromptServer.instance.send_sync(
            EVENT_NAME,  # This is our custom event name
            {
                'action': action,
                'arg1': arg1,
                'arg2': arg2
            },
            sid
        )
    except Exception as e:
        logger.error(f"when trying to use ComfyUI PromptServer: {e}")
