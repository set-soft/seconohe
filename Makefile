#!/usr/bin/make

DOCS_DIR := ./docs
EXCLUDED = _ansi foreground_estimation.fmlfe_cupy foreground_estimation.fmlfe_numba foreground_estimation.fmlfe_opencl
EXCLUDED_V = $(addprefix -e seconohe., $(EXCLUDED))

.PHONY: docs clean-docs

docs:
	@echo "Generating documentation with pdoc..."
	tool/build_docs.py $(EXCLUDED_V)  -o $(DOCS_DIR) -t ./resources/pdoc_dark --override-var 'JS_PATH:Path to JS extensions' seconohe
	@echo "Documentation generated in $(DOCS_DIR)/"

clean-docs:
	@echo "Cleaning documentation directory..."
	@rm -rf $(DOCS_DIR)
