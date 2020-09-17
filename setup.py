from setuptools import setup

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
    version="0.1.1",
    install_requires=[
        "pytz",
        "pandas>=1.0",
        "numpy",
        "pyerf",
        "SQLAlchemy",
        "psycopg2-binary",
        "isodate",
        "openturns",
        "properscoring",
        "altair==3.0.0",
        "selenium",
    ],
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    packages=[
        "timely_beliefs",
        "timely_beliefs.beliefs",
        "timely_beliefs.sensors",
        "timely_beliefs.sensors.func_store",
        "timely_beliefs.sources",
        "timely_beliefs.examples",
        "timely_beliefs.visualization",
    ],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
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
