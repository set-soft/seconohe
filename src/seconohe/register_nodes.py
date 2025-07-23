# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnologïa Industrial
# License: GPL-3.0
# Project: SeCoNoHe
# Helper to register all nodes from the provided modules
import inspect
import logging
import types
from typing import List


def register_nodes(logger: logging.Logger, modules: List[types.ModuleType]):
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

    for m in modules:
        suffix = " " + m.SUFFIX if hasattr(m, "SUFFIX") else ""
        if suffix:
            suffix = " " + suffix
        for name, obj in inspect.getmembers(m):
            # We skip nodes imported from the ComfyUI main nodes
            if not inspect.isclass(obj) or not hasattr(obj, "INPUT_TYPES") or obj.__module__ == "nodes":
                continue
            assert hasattr(obj, "UNIQUE_NAME"), f"No name for {obj.__name__}"
            NODE_CLASS_MAPPINGS[obj.UNIQUE_NAME] = obj
            NODE_DISPLAY_NAME_MAPPINGS[obj.UNIQUE_NAME] = obj.DISPLAY_NAME + suffix

    logger.info(f"Registering {len(NODE_CLASS_MAPPINGS)} node(s).")
    logger.debug(f"{list(NODE_DISPLAY_NAME_MAPPINGS.values())}")

    return NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
