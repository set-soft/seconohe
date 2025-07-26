# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnologïa Industrial
# License: GPL-3.0
# Project: SeCoNoHe
# Helper to register all nodes from the provided modules
import inspect
import logging
import types
from typing import List, Tuple, Type, Optional


def register_nodes(
    logger: logging.Logger,
    modules: List[types.ModuleType],
    version: Optional[str] = None
) -> Tuple[dict[str, Type], dict[str, str]]:
    """
    Scans a list of modules to discover and register ComfyUI node classes.

    This function iterates through the provided Python modules, inspects their
    members, and identifies classes that conform to the ComfyUI node structure.
    It then populates and returns the mapping dictionaries required by ComfyUI.

    Node classes are expected to have the following class attributes:
    - ``INPUT_TYPES`` (classmethod)
    - ``RETURN_TYPES`` (tuple)
    - ``FUNCTION`` (str)
    - ``CATEGORY`` (str)
    - ``UNIQUE_NAME`` (str): The internal name for the node.
    - ``DISPLAY_NAME`` (str, optional): The user-facing name in the UI.

    Additionally, a module can define a ``SUFFIX`` string attribute, which will
    be appended to the display name of all nodes found in that module.

    :param logger: A logger instance for logging registration activity.
    :type logger: logging.Logger
    :param modules: A list of Python module objects to scan for nodes.
    :type modules: List[types.ModuleType]
    :param version: The version of the nodes, when provided is included in the logs.
    :type version: Optional[str]
    :return: A tuple containing NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS.
    :rtype: Tuple[dict[str, Type], dict[str, str]]
    """
    node_class_mappings: dict[str, Type] = {}
    node_display_name_mappings: dict[str, str] = {}

    logger.debug(f"Scanning {len(modules)} module(s) for nodes...")
    for module in modules:
        # Optional suffix for display names
        suffix = f" {module.SUFFIX}" if hasattr(module, "SUFFIX") else ""

        logger.debug(f"Scanning module '{module.__name__}' with suffix '{suffix}'")
        for name, obj in inspect.getmembers(module):
            # Skip anything that's not a class
            if not inspect.isclass(obj):
                continue

            # Ensures we only register classes defined IN THIS MODULE, not imported ones.
            if obj.__module__ != module.__name__:
                continue

            # Check this is a complete node
            if not (hasattr(obj, 'INPUT_TYPES') and hasattr(obj, 'RETURN_TYPES') and
                    hasattr(obj, 'FUNCTION') and hasattr(obj, 'CATEGORY')):
                continue

            # Inform about missing UNIQUE_NAME
            if not hasattr(obj, 'UNIQUE_NAME'):
                logger.warning(f"Class `{obj.__name__}` looks like a node but is missing the "
                               "required `UNIQUE_NAME` class attribute. Skipping.")
                continue

            unique_name = obj.UNIQUE_NAME

            # DISPLAY_NAME optional with a sensible fallback.
            display_name = getattr(obj, 'DISPLAY_NAME', unique_name)

            if unique_name in node_class_mappings:
                logger.warning(f"Duplicate UNIQUE_NAME '{unique_name}' detected! "
                               f"Class `{obj.__name__}` will overwrite the existing entry.")

            node_class_mappings[unique_name] = obj
            node_display_name_mappings[unique_name] = display_name + suffix

    version_str = f" for version {version}" if version is not None else ""
    logger.info(f"Registering {len(node_class_mappings)} node(s){version_str}.")
    logger.debug(f"Registered display names: {list(node_display_name_mappings.values())}")

    return node_class_mappings, node_display_name_mappings
