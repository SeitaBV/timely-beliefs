# Note: use tabs
# actions which are virtual, i.e. not a script
.PHONY: install-deps install-tb freeze-deps test

install: install-deps install-tb

install-deps:
	pip install --upgrade pip-tools
	pip-sync dev/requirements.txt
	pip install pre-commit

freeze-deps:
	pip install --upgrade pip-tools
	pip-compile -o dev/requirements.txt	 # use --upgrade or --upgrade-package to actually change versions

install-tb:
	pip install -e .
	pre-commit install

test:
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
