# Makefile to help automate key steps

.DEFAULT_GOAL := help
# Will likely fail on Windows, but Makefiles are in general not Windows
# compatible so we're not too worried
TEMP_FILE := $(shell mktemp)

# A helper script to get short descriptions of each target in the Makefile
define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([\$$\(\)a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-30s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT


help:  ## print short description of each target
	@python3 -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

.PHONY: checks
checks:  ## run all the linting checks of the codebase
	@echo "=== pre-commit ==="; uv run pre-commit run --all-files || echo "--- pre-commit failed ---" >&2; \
		echo "=== mypy ==="; MYPYPATH=stubs uv run mypy src || echo "--- mypy failed ---" >&2; \
		echo "======"

.PHONY: ruff-fixes
ruff-fixes:  ## fix and format the code using ruff
	uv run ruff check --fix src tests scripts docs/source/conf.py docs/source/notebooks/*.py
	uv run ruff format src tests scripts docs/source/conf.py docs/source/notebooks/*.py

.PHONY: test
test:  ## run the tests
	uv run pytest src tests -r a -v --doctest-modules --cov=src

.PHONY: docs
docs:  ## build the docs
	uv run sphinx-build -T -b html docs/source docs/build/html

.PHONY: licence-check
licence-check:  ## Check that licences of the dependencies are suitable
	# Will likely fail on Windows, but Makefiles are in general not Windows
	# compatible so we're not too worried
	./scripts/check-licences.sh

.PHONY: virtual-environment
virtual-environment:  ## update virtual environment, create a new one if it doesn't already exist
	uv sync --all-extras
	uv run pre-commit install

	# Set jupytext as the default viewer when opening notebooks
	uv run jupytext-config set-default-viewer
