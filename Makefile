# Check Python major and minor version
# For more information, see https://stackoverflow.com/a/22105036
PYV = $(shell python -c "import sys;t='{v[0]}.{v[1]}'.format(v=list(sys.version_info[:2]));sys.stdout.write(t)")

# Note: use tabs (not spaces) for indentation
# actions which are virtual, i.e. not a script
.PHONY: install install-deps install-tb install-for-test freeze-deps upgrade-deps test test-core test-forecast test-viz ensure-deps-folder

install: install-deps install-tb

install-for-test:
	pip install setuptools_scm pytest

install-deps:
	pip install --upgrade pip-tools
	pip-sync dev/${PYV}/requirements.txt
	pip install pre-commit

install-tb:
	pip install -e .
	pre-commit install

freeze-deps:
	make ensure-deps-folder
	pip install --upgrade pip-tools
	pip-compile -o dev/${PYV}/requirements.txt

upgrade-deps:
	make ensure-deps-folder
	pip install --upgrade pip-tools
	pip-compile -o dev/${PYV}/requirements.txt --upgrade
ifneq ($(skip-test), yes)
	make test
endif

test:
	make test-core  # test just the core, which may have fewer pinned dependencies
	make test-all  # test everything sharing the same dependencies

test-all:
	pip install -e .[forecast,viz]
	make install-for-test
	pytest

test-core:
	pip install -e .
	make install-for-test
	pytest --ignore test_forecast__ --ignore test_viz__

test-forecast:
	pip install -e .[forecast]
	make install-for-test
	pytest -k test_forecast__

test-viz:
	pip install -e .[viz]
	make install-for-test
	pytest -k test_viz__

ensure-deps-folder:
	mkdir -p dev/${PYV}
