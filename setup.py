from setuptools import find_packages, setup


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
