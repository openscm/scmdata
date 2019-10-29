.DEFAULT_GOAL := help

VENV_DIR ?= ./venv

FILES_TO_FORMAT_PYTHON=setup.py scripts src tests docs/source/conf.py

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

.PHONY: help
help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

test:  $(VENV_DIR) ## run the full testsuite
	$(VENV_DIR)/bin/pytest --cov -r a --cov-report term-missing

.PHONY: format
format:  ## re-format files
	make isort
	make black

.PHONY: flake8
flake8: $(VENV_DIR)  ## check compliance with pep8
	$(VENV_DIR)/bin/flake8 $(FILES_TO_FORMAT_PYTHON)

.PHONY: isort
isort: $(VENV_DIR)  ## format the imports in the source and tests
	$(VENV_DIR)/bin/isort -y --recursive $(FILES_TO_FORMAT_PYTHON)

.PHONY: black
black: $(VENV_DIR)  ## use black to autoformat code
	@status=$$(git status --porcelain); \
	if test "x$${status}" = x; then \
		$(VENV_DIR)/bin/black --exclude _version.py --target-version py37 $(FILES_TO_FORMAT_PYTHON); \
	else \
		echo Not trying any formatting, working directory is dirty... >&2; \
	fi;

virtual-environment:  ## update venv, create a new venv if it doesn't exist
	make $(VENV_DIR)

$(VENV_DIR): setup.py
	[ -d $(VENV_DIR) ] || python3 -m venv $(VENV_DIR)

	$(VENV_DIR)/bin/pip install --upgrade pip
	$(VENV_DIR)/bin/pip install -e .[dev]

	touch $(VENV_DIR)


# first time setup, follow this https://blog.jetbrains.com/pycharm/2017/05/how-to-publish-your-package-on-pypi/
# then this works
.PHONY: publish-on-testpypi
publish-on-testpypi: $(VENV_DIR)  ## publish the current state of the repository to test PyPI
	-rm -rf build dist
	@status=$$(git status --porcelain); \
	if test "x$${status}" = x; then \
		$(VENV_DIR)/bin/python setup.py sdist bdist_wheel --universal; \
		$(VENV_DIR)/bin/twine upload --verbose -r testpypi dist/*; \
	else \
		echo Working directory is dirty >&2; \
	fi;

.PHONY: test-testpypi-install
test-testpypi-install: $(VENV_DIR)  ## test whether installing from test PyPI works
	$(eval TEMPVENV := $(shell mktemp -d))
	python3 -m venv $(TEMPVENV)
	$(TEMPVENV)/bin/pip install pip --upgrade
	# Install pymagicc without dependencies.
	$(TEMPVENV)/bin/pip install \
		-i https://testpypi.python.org/pypi scmdata \
		--no-dependencies --pre
		# Remove local directory from path to get actual installed version.
	@echo "This doesn't test all dependencies"
	$(TEMPVENV)/bin/python -c "import sys; sys.path.remove(''); import scmdata; print(scmdata.__version__)"

.PHONY: publish-on-pypi
publish-on-pypi:  $(VENV_DIR) ## publish the current state of the repository to PyPI
	-rm -rf build dist
	@status=$$(git status --porcelain); \
	if test "x$${status}" = x; then \
		$(VENV_DIR)/bin/python setup.py sdist bdist_wheel --universal; \
		$(VENV_DIR)/bin/twine upload --verbose dist/*; \
	else \
		echo Working directory is dirty >&2; \
	fi;

.PHONY: test-pypi-install
test-pypi-install: $(VENV_DIR)  ## test whether installing from PyPI works
	$(eval TEMPVENV := $(shell mktemp -d))
	python3 -m venv $(TEMPVENV)
	$(TEMPVENV)/bin/pip install pip --upgrade
	$(TEMPVENV)/bin/pip install scmdata --pre
	$(TEMPVENV)/bin/python scripts/test_install.py

.PHONY: check-pypi-distribution
check-pypi-distribution: $(VENV_DIR)  ## check the PyPI distribution for errors
	-rm -rf build dist
	@status=$$(git status --porcelain); \
	if test "x$${status}" = x; then \
		$(VENV_DIR)/bin/python setup.py sdist bdist_wheel --universal; \
		$(VENV_DIR)/bin/twine check dist/*; \
	else \
		echo Working directory is dirty >&2; \
	fi;

.PHONY: test-install
test-install: $(VENV_DIR)  ## test whether installing the local setup works
	$(eval TEMPVENV := $(shell mktemp -d))
	python3 -m venv $(TEMPVENV)
	$(TEMPVENV)/bin/pip install pip --upgrade
	$(TEMPVENV)/bin/pip install .
	$(TEMPVENV)/bin/python scripts/test_install.py
