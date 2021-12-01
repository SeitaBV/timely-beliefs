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
	python setup.py develop
	pre-commit install

test:
	python setup.py develop
	pip install setuptools_scm pytest
	pytest
