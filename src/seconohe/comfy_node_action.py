# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
# License: GPLv3
# Project: SeCoNoHe
#
# ComfyUI Node actions
import logging
from typing import Optional, Any, Literal
# ComfyUI imports
try:
    from server import PromptServer
    with_comfy = True
except Exception:
    with_comfy = False

_EVENT_NAME = "seconohe-node"
# Define the known actions as a Literal for better type checking and auto-completion
NodeAction = Literal["change_widget"]


def send_node_action(logger: logging.Logger, action: str, arg1: Any = None, arg2: Any = None,
                     sid: Optional[str] = None) -> None:
    """
    Sends a custom event to the frontend to perform an action on a node.

    This function communicates with a corresponding JavaScript listener for the
    ``seconohe-node`` event, instructing it to perform a specific action,
    such as changing a widget's value. It fails silently if not run within an
    active ComfyUI server environment.

    **Known Actions:**

    - **``change_widget``**:
      - ``arg1`` (str): The name (label) of the widget to change.
      - ``arg2`` (Any): The new value for the widget. The type should
        match what the widget expects (e.g., str, int, float).

    Example Usage::

        # To change a widget named 'seed' to the value 12345
        send_node_action(logger, 'change_widget', 'seed', 12345)

    :param logger: The logger instance for reporting errors.
    :type logger: logging.Logger
    :param action: The name of the action for the frontend to perform.
    :type action: Literal['change_widget']
    :param arg1: The first argument for the action. Its meaning depends on `action`.
                 Defaults to ``None``.
    :type arg1: Any
    :param arg2: The second argument for the action. Its meaning depends on `action`.
                 Defaults to ``None``.
    :type arg2: Any
    :param sid: The session ID of a specific client. If ``None``, broadcasts to
                all clients. Defaults to ``None``.
    :type sid: Optional[str]
    """
    # Check if we are in a ComfyUI environment and the server instance is available
    if not with_comfy or not hasattr(PromptServer, 'instance') or PromptServer.instance is None:
        logger.debug("Send node action skipped: Not running in an active ComfyUI server environment.")
        return
    try:
        payload = {
            'action': action,
            'arg1': arg1,
            'arg2': arg2
        }
        PromptServer.instance.send_sync(_EVENT_NAME, payload, sid)
        logger.debug(f"Sent node action: {payload}")
    except Exception as e:
        logger.error(f"Failed to send node action via ComfyUI PromptServer: {e}")
