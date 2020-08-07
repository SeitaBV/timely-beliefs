# Note: use tabs
# actions which are virtual, i.e. not a script
.PHONY: install-deps install-tb freeze-deps test

install: install-deps install-tb

install-deps:
	pip install pip-tools
	pip-sync

freeze-deps:
	pip install pip-tools
	pip-compile  # use --upgrade or --upgrade-package to actually change versions

install-tb:
	python setup.py develop

test:
	pip install pytest
	pytest
