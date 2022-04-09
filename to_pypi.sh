#!/bin/bash


# Script to release Timely-Beliefs to PyPi.
#
# Cleans up build and dist dirs, checks for python files which are not in git, installs dependencies
# and finally uploads tar and wheel packages to Pypi.
#
#
# Usage
# ------------
#
# ./to_pypi [--dry-run]
#
# If the --dry-run flag is present, this script will do all steps, but skip the upload to Pypi.
# 
#
# The version
# -------------
# The version comes from setuptools_scm. See `python setup.py --version`.
# setuptools_scm works via git tags that should implement a semantic versioning scheme, e.g. v0.2.3
#
# If there were zero commits since the most recent tag, we have a real release and the version basically *is* what the tag says.
# Otherwise, the version also includes a .devN identifier, where N is the number of commits since the last version tag.
# If you want, you can read more about acceptable versions in PEP 440: https://www.python.org/dev/peps/pep-0440/


echo "[TO_PYPI] Cleaning ..."
rm -rf build/* dist/*

echo "[TO_PYPI] Installing dependencies ..."
pip -q install twine
pip -q install wheel

echo "[TO_PYPI] Packaging ..."
python setup.py egg_info sdist
python setup.py egg_info bdist_wheel

if [ "$1" == "--dry-run" ]; then
    echo "[TO_PYPI] Not uploading to Pypi (--dry-run active) ..."
    exit
fi
echo "[TO_PYPI] Uploading to Pypi ..."
twine upload dist/*
