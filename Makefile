#!/usr/bin/make

DOCS_DIR := ./docs

.PHONY: docs clean-docs

docs:
	@echo "Generating documentation with pdoc..."
	tool/build_docs.py -e seconohe._ansi -o $(DOCS_DIR) -t ./resources/pdoc_dark --override-var 'JS_PATH:Path to JS extensions' seconohe
	@echo "Documentation generated in $(DOCS_DIR)/"

clean-docs:
	@echo "Cleaning documentation directory..."
	@rm -rf $(DOCS_DIR)
