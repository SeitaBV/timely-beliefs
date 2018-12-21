from setuptools import setup

setup(
    name="timely-beliefs",
    description="Data modelled as beliefs (at a certain time) about events (at a certain time).",
    author="Seita BV",
    author_email="felix@seita.nl",
    keywords=["time series", "uncertainty", "lineage"],
    version="0.0.1",
    install_requires=[
        "pandas",
        "numpy",
        "SQLAlchemy",
        "Flask-SQLAlchemy",
    ],
    tests_require=["pytest"],
    packages=["timely_beliefs"],
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
