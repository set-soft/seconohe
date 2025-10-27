# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnologïa Industrial
# License: GPL-3.0
# Project: SeCoNoHe
# Helper to register all nodes from the provided modules
import inspect
import logging
import os
import types
from typing import List, Tuple, Type, Optional
try:
    from comfy_api.latest import ComfyExtension, io
    comfy_v3_available = True
except ImportError:
    comfy_v3_available = False

    # Dummies for the docs generation
    class ComfyExtension():
        pass

    class io():
        class ComfyNode():
            pass
try:
    # ComfyUI folders code
    import folder_paths
except ImportError:
    pass


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


class ComfyReg(ComfyExtension):
    """ This is the type that must be returned by `comfy_entrypoint` """
    def __init__(self, classes):
        super().__init__()
        self.classes = classes

    # Must be declared as async
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return self.classes


def register_nodes_v3(
    logger: logging.Logger,
    modules: List[types.ModuleType],
    version: Optional[str] = None
) -> ComfyExtension:
    """
    Scans a list of modules to discover and register ComfyUI node classes.
    This is for the API V3.

    This function iterates through the provided Python modules, inspects their
    members, and identifies classes that conform to the ComfyUI node structure.
    It creates the registering object required by ComfyUI.

    Node classes are expected to have the following class attributes:
    - ``define_schema`` (classmethod)
    - ``execute`` (classmethod)

    :param logger: A logger instance for logging registration activity.
    :type logger: logging.Logger
    :param modules: A list of Python module objects to scan for nodes.
    :type modules: List[types.ModuleType]
    :param version: The version of the nodes, when provided is included in the logs.
    :type version: Optional[str]
    :return: A `ComfyExtension` object suitable for the `comfy_entrypoint`
    :rtype: ComfyExtension
    """
    v3_nodes: Type = []

    logger.debug(f"Scanning {len(modules)} module(s) for nodes...")
    for module in modules:
        logger.debug(f"Scanning module '{module.__name__}'")
        for name, obj in inspect.getmembers(module):
            # Skip anything that's not a class
            if not inspect.isclass(obj):
                continue

            # Ensures we only register classes defined IN THIS MODULE, not imported ones.
            if obj.__module__ != module.__name__:
                continue

            if not (hasattr(obj, 'define_schema') and hasattr(obj, 'execute')):
                continue

            v3_nodes.append(obj)

    version_str = f" for version {version}" if version is not None else ""
    logger.info(f"Registering {len(v3_nodes)} node(s){version_str}.")
    logger.debug(f"Registered classes: {[c.__name__ for c in v3_nodes]}")

    return ComfyReg(v3_nodes)


def register_models_key(logger: logging.Logger, key: str, dir: str, extra_keys: Optional[dict[str, str]] = None) -> None:
    """
    Registers a new folder key `key` pointing to `dir`.
    If the key is already registered we add `dir` to the current key.

    When registering a new key and `extra_keys` is provided we also
    merge the dirs from these keys to our dirs.

    :param logger: A logger instance for logging registration activity.
    :type logger: logging.Logger
    :param key: A folder key (like ones in `extra_model_paths.yaml`)
    :type key: str
    :param dir: The dir to use for this key (ComfyUI/models/{dir})
    :type dir: str
    :param extra_keys: Other keys to merge to ours
    :type extra_keys: Optional[dict[str, str]]
    """
    # ComfyUI/models/{dir} will be the default
    models_dir_default = os.path.join(folder_paths.models_dir, dir)
    try:
        if not os.path.exists(models_dir_default):
            os.makedirs(models_dir_default, exist_ok=True)  # Ensure the dir exists
    except Exception:
        # Read-only file system?
        pass
    # Add our key
    folder_paths.add_model_folder_path(key, models_dir_default)
    logger.debug(f"Registered folder `{models_dir_default}` with key `{key}`")
    exts = folder_paths.supported_pt_extensions
    # Merge keys
    if extra_keys:
        for k, v in extra_keys.items():
            m_dirs, m_exts = folder_paths.folder_names_and_paths.get(k, ([os.path.join(folder_paths.models_dir, v)], set()))
            for dir in m_dirs:
                folder_paths.add_model_folder_path(key, dir)
                logger.debug(f"- Adding {dir}")
            exts |= m_exts
    # Now update the extensions
    m_dirs, m_exts = folder_paths.folder_names_and_paths[key]
    folder_paths.folder_names_and_paths[key] = (m_dirs, exts)
    logger.debug(f"Final dirs: {m_dirs}")
    logger.debug(f"Final exts: {exts}")


def check_v3(logger: logging.Logger) -> None:
    """ Check if ComfyUI API v3 is implemented, otherwise raise an error

    :param logger: A logger instance.
    :type logger: logging.Logger
    """
    if comfy_v3_available:
        return
    msg = "ComfyUI 0.3.48 (Aug 1, 2025) is needed, please upgrade"
    logger.error(msg)
    raise ValueError(msg)
