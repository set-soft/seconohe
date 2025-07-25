#!/usr/bin/env python3
# build_docs.py
"""
A command-line tool to generate documentation for a Python project using pdoc.

This script uses pdoc as a library to gain fine-grained control over the
documentation output, such as modifying variable representations and excluding
private modules.
"""

import argparse
from pathlib import Path
from pdoc.doc import Module
from pdoc.render import configure
from pdoc import pdoc
import logging
from typing import List, Dict, Optional, Tuple

# Set up a simple logger for this script
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def build_documentation(
    module_name: str,
    output_dir: Path,
    exclude_list: List[str],
    template_dir: Optional[Path],
    var_overrides: Dict[str, str]
) -> int:
    """
    Generates documentation for a Python module using pdoc's library interface.

    Args:
        module_name (str): The name of the Python package/module to document.
        output_dir (Path): The directory where the HTML documentation will be saved.
        exclude_list (List[str]): A list of modules to exclude, prefixed with '!'.
        template_dir (Optional[Path]): Path to a custom pdoc template directory.
        var_overrides (Dict[str, str]): A dict mapping variable names to new
                                        string representations for the docs.

    Returns:
        int: Exit code (0 for success, 1 for failure).
    """
    logger.info(f"Generating documentation for module: '{module_name}'")

    try:
        module_doc = Module.from_name(module_name)
    except ImportError:
        logger.error(f"Could not find module '{module_name}'. Make sure it is installed or in your PYTHONPATH.")
        logger.error("Try running 'pip install -e .' from your project root.")
        return 1

    if var_overrides:
        logger.info("Applying variable representation overrides...")
        for var_path, new_value_str in var_overrides.items():
            try:
                var_doc_obj = module_doc.get(var_path)
                var_doc_obj.default_value_str = new_value_str
                logger.info(f"  - Overriding '{var_path}' representation.")
            except KeyError:
                logger.warning(f"  - Could not find variable '{var_path}' to override. Skipping.")

    if template_dir and template_dir.is_dir():
        logger.info(f"Using custom template from: {template_dir}")
        configure(template_directory=template_dir)
    elif template_dir:
        logger.warning(f"Template directory not found: {template_dir}. Using pdoc default.")

    logger.info(f"Excluding modules: {exclude_list}")
    logger.info(f"Writing HTML output to: {output_dir}")
    pdoc(module_name, *exclude_list, output_directory=output_dir)
    logger.info("Documentation generated successfully.")
    return 0


def parse_var_override(value: str) -> Tuple[str, str]:
    """Custom argparse type to parse NAME:VALUE strings."""
    if ':' not in value:
        raise argparse.ArgumentTypeError(
            f"Invalid format for variable override: '{value}'. "
            "Expected format is 'VARIABLE_NAME:New String Representation'."
        )
    name, new_value = value.split(':', 1)
    if not name:
        raise argparse.ArgumentTypeError("Variable name cannot be empty.")
    return name, new_value


def main() -> int:
    """Main function to parse arguments and run the documentation build."""
    parser = argparse.ArgumentParser(
        description="Generate project documentation using pdoc.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "module",
        nargs='?',
        default="seconohe",
        help="The name of the Python package/module to document."
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path("./docs"),
        help="The directory where the HTML documentation will be saved."
    )
    parser.add_argument(
        "-e", "--exclude",
        action="append",
        default=[],
        help="A module to exclude from documentation (e.g., 'seconohe._ansi'). Can be specified multiple times."
    )
    parser.add_argument(
        "-t", "--template-dir",
        type=Path,
        default=None,
        help="Path to a custom pdoc template directory. If not specified, pdoc's default is used."
    )
    parser.add_argument(
        "--override-var",
        type=parse_var_override,
        action="append",
        default=[],
        dest="var_overrides",
        help="Override a variable's string representation in the docs. "
             "Format: 'VAR_NAME:New String Value'. Can be specified multiple times."
    )

    args = parser.parse_args()

    # Prepend '!' to all user-provided exclusion strings
    exclude_list_with_prefix = [f"!{mod}" for mod in args.exclude]

    # Convert list of tuples from argparse into a dictionary
    variable_overrides_dict = dict(args.var_overrides)

    exit_code = build_documentation(
        module_name=args.module,
        output_dir=args.output_dir,
        exclude_list=exclude_list_with_prefix,
        template_dir=args.template_dir,
        var_overrides=variable_overrides_dict
    )

    return exit_code


if __name__ == "__main__":
    import sys
    sys.exit(main())
