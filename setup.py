#!/usr/bin/env python
"""ESMValTool installation script."""
# This script only installs dependencies available on PyPI
#
# Dependencies that need to be installed some other way (e.g. conda):
# - ncl
# - iris
# - python-stratify

import json
import os
import re
import sys
from pathlib import Path

from setuptools import Command, setup

PACKAGES = [
    "esmvalcore",
]

REQUIREMENTS = {
    # Installation script (this file) dependencies
    "setup": [
        "setuptools_scm",
    ],
    # Installation dependencies
    # Use with pip install . to install from source
    "install": [
        "cartopy",
        "cf-units",
        "dask[array,distributed]!=2024.8.0",  # ESMValCore/issues/2503
        "dask-jobqueue",
        "esgf-pyclient>=0.3.1",
        "esmf-regrid>=0.11.0",
        "esmpy!=8.1.0",  # not on PyPI
        "filelock",
        "fiona",
        "fire",
        "geopy",
        "humanfriendly",
        "iris-grib>=0.20.0",  # github.com/ESMValGroup/ESMValCore/issues/2535
        "isodate",
        "jinja2",
        "nc-time-axis",  # needed by iris.plot
        "nested-lookup",
        "netCDF4",
        "numpy!=1.24.3,<2.0.0",  # avoid pulling 2.0.0rc1
        "packaging",
        "pandas",
        "pillow",
        "prov",
        "psutil",
        "py-cordex",
        "pybtex",
        "pyyaml",
        "requests",
        "scipy>=1.6",
        "scitools-iris>=3.10.0",
        "shapely>=2.0.0",
        "stratify>=0.3",
        "yamale",
    ],
    # Test dependencies
    "test": [
        "pytest>=3.9,!=6.0.0rc1,!=6.0.0",
        "pytest-cov>=2.10.1",
        "pytest-env",
        "pytest-html!=2.1.0",
        "pytest-metadata>=1.5.1",
        "pytest-mock",
        "pytest-xdist",
        "ESMValTool_sample_data==0.0.3",
    ],
    # Documentation dependencies
    "doc": [
        "autodocsumm>=0.2.2",
        "ipython",
        "nbsphinx",
        "sphinx>=6.1.3",
        "pydata_sphinx_theme",
    ],
    # Development dependencies
    # Use pip install -e .[develop] to install in development mode
    "develop": [
        "pre-commit",
        "pylint",
        "pydocstyle",
        "vprof",
    ],
}


def discover_python_files(paths, ignore):
    """Discover Python files."""

    def _ignore(path):
        """Return True if `path` should be ignored, False otherwise."""
        return any(re.match(pattern, path) for pattern in ignore)

    for path in sorted(set(paths)):
        for root, _, files in os.walk(path):
            if _ignore(path):
                continue
            for filename in files:
                filename = os.path.join(root, filename)
                if filename.lower().endswith(".py") and not _ignore(filename):
                    yield filename


class CustomCommand(Command):
    """Custom Command class."""

    def install_deps_temp(self):
        """Try to temporarily install packages needed to run the command."""
        if self.distribution.install_requires:
            self.distribution.fetch_build_eggs(
                self.distribution.install_requires
            )
        if self.distribution.tests_require:
            self.distribution.fetch_build_eggs(self.distribution.tests_require)


class RunLinter(CustomCommand):
    """Class to run a linter and generate reports."""

    user_options: list = []

    def initialize_options(self):
        """Do nothing."""

    def finalize_options(self):
        """Do nothing."""

    def run(self):
        """Run prospector and generate a report."""
        check_paths = PACKAGES + [
            "setup.py",
            "tests",
        ]
        ignore = [
            "doc/",
        ]

        # try to install missing dependencies and import prospector
        try:
            from prospector.run import main
        except ImportError:
            # try to install and then import
            self.distribution.fetch_build_eggs(["prospector[with_pyroma]"])
            from prospector.run import main

        self.install_deps_temp()

        # run linter

        # change working directory to package root
        package_root = os.path.abspath(os.path.dirname(__file__))
        os.chdir(package_root)

        # write command line
        files = discover_python_files(check_paths, ignore)
        sys.argv = ["prospector"]
        sys.argv.extend(files)

        # run prospector
        errno = main()

        sys.exit(errno)


def read_authors(filename):
    """Read the list of authors from .zenodo.json file."""
    with Path(filename).open(encoding="utf-8") as file:
        info = json.load(file)
        authors = []
        for author in info["creators"]:
            name = " ".join(author["name"].split(",")[::-1]).strip()
            authors.append(name)
        return ", ".join(authors)


def read_description(filename):
    """Read the description from .zenodo.json file."""
    with Path(filename).open(encoding="utf-8") as file:
        info = json.load(file)
        return info["description"]


setup(
    name="ESMValCore",
    author=read_authors(".zenodo.json"),
    description=read_description(".zenodo.json"),
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://www.esmvaltool.org",
    download_url="https://github.com/ESMValGroup/ESMValCore",
    license="Apache License, Version 2.0",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Hydrology",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    packages=PACKAGES,
    # Include all version controlled files
    include_package_data=True,
    setup_requires=REQUIREMENTS["setup"],
    install_requires=REQUIREMENTS["install"],
    tests_require=REQUIREMENTS["test"],
    extras_require={
        "develop": REQUIREMENTS["develop"]
        + REQUIREMENTS["test"]
        + REQUIREMENTS["doc"],
        "test": REQUIREMENTS["test"],
        "doc": REQUIREMENTS["doc"],
    },
    entry_points={
        "console_scripts": [
            "esmvaltool = esmvalcore._main:run",
        ],
    },
    cmdclass={
        "lint": RunLinter,
    },
    zip_safe=False,
)
