import sys

from setuptools import find_packages, setup


def get_numpy_version():
    """numpy's setup requires minimal Python versions"""
    if sys.version_info[:2] <= (3, 6):
        return "numpy==1.19.5"
    if sys.version_info[:2] <= (3, 7):
        return "numpy==1.21.4"
    return "numpy"


def get_scipy_version():
    """scipy's setup requires minimal Python versions"""
    if sys.version_info[:2] <= (3, 6):
        return "scipy<1.6"
    if sys.version_info[:2] == (3, 7):
        return "scipy<1.8"
    return "scipy"


setup(
    name="timely-beliefs",
    description="Data modelled as beliefs (at a certain time) about events (at a certain time).",
    author="Seita BV",
    author_email="felix@seita.nl",
    url="https://github.com/seitabv/timely-beliefs",
    keywords=[
        "time series",
        "forecasting",
        "analytics",
        "visualization",
        "uncertainty",
        "lineage",
    ],
    python_requires=">=3.8",  # not enforced, just info. 3.6 and 3.7 are possible to install, but we don't support
    install_requires=[
        "importlib_metadata",
        "pytz",
        # https://pandas.pydata.org/pandas-docs/stable/whatsnew/v1.2.0.html#increased-minimum-version-for-python
        "pandas>=1.1.5,<1.2"
        if sys.version_info[:2] == (3, 6)
        else "pandas>=1.1.5,<1.3",
        get_numpy_version(),
        get_scipy_version(),
        "SQLAlchemy",
        "psycopg2-binary",
        "isodate",
        "openturns",
        "properscoring",
        "altair>=4.0.0",
        "selenium",
    ],
    setup_requires=["setuptools_scm"],
    use_scm_version={"local_scheme": "no-local-version"},  # handled by setuptools_scm
    packages=find_packages(),
    include_package_data=True,  # now setuptools_scm adds all files under source control
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    long_description="""\
    Model to represent data as beliefs about events, stored in the form of
    a multi-index pandas DataFrame enriched with attributes to get out convenient representations of the data.
    """,
)
