# Check Python major and minor version
# For more information, see https://stackoverflow.com/a/22105036
PYV = $(shell python -c "import sys;t='{v[0]}.{v[1]}'.format(v=list(sys.version_info[:2]));sys.stdout.write(t)")

# Note: use tabs (not spaces) for indentation
# actions which are virtual, i.e. not a script
.PHONY: install install-deps install-tb freeze-deps upgrade-deps test test-core test-forecast test-viz

install: install-deps install-tb

install-deps:
	pip install --upgrade pip-tools
	pip-sync dev/${PYV}/requirements.txt
	pip install pre-commit

install-tb:
	pip install -e .
	pre-commit install

freeze-deps:
	pip install --upgrade pip-tools
	pip-compile -o dev/${PYV}/requirements.txt	 # use --upgrade or --upgrade-package to actually change versions

upgrade-deps:
	pip install --upgrade pip-tools
	pip-compile -o dev/${PYV}/requirements.txt --upgrade
	make test

test:
	make test-core
	make test-forecast
	make test-viz

test-core:
	pip install -e .
	pip install setuptools_scm pytest
	pytest --ignore test_forecast__ --ignore test_viz__

test-forecast:
	pip install -e .[forecast]
	pip install setuptools_scm pytest
	pytest -k test_forecast__

test-viz:
	pip install -e .[viz]
	pip install setuptools_scm pytest
	pytest -k test_viz__
