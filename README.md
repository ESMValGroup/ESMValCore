# ESMValCore package

[![Documentation Status](https://readthedocs.org/projects/esmvaltool/badge/?version=latest)](https://esmvaltool.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3387139.svg)](https://doi.org/10.5281/zenodo.3387139)
[![Gitter](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/ESMValGroup?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![CircleCI](https://circleci.com/gh/ESMValGroup/ESMValCore.svg?style=svg)](https://circleci.com/gh/ESMValGroup/ESMValCore)
[![Codacy Badge](https://api.codacy.com/project/badge/Coverage/5d496dea9ef64ec68e448a6df5a65783)](https://www.codacy.com/app/ESMValGroup/ESMValCore?utm_source=github.com&utm_medium=referral&utm_content=ESMValGroup/ESMValCore&utm_campaign=Badge_Coverage)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/5d496dea9ef64ec68e448a6df5a65783)](https://www.codacy.com/app/ESMValGroup/ESMValCore?utm_source=github.com&utm_medium=referral&utm_content=ESMValGroup/ESMValCore&utm_campaign=Badge_Grade)
[![Docker Build Status](https://img.shields.io/docker/build/esmvalgroup/esmvaltool.svg)](https://hub.docker.com/r/esmvalgroup/esmvaltool/)
[![Anaconda-Server Badge](https://anaconda.org/esmvalgroup/esmvalcore/badges/installer/conda.svg)](https://conda.anaconda.org/esmvalgroup)

ESMValCore: core functionalities for the ESMValTool, a community
diagnostic and performance metrics tool for routine evaluation of Earth System
Models in CMIP.

# Getting started

## Installing from Anaconda

The Anaconda package can be found on [ESMValGroup Anaconda Channel.](https://anaconda.org/ESMValGroup)

If you already have
[Anaconda installed](https://conda.io/projects/conda/en/latest/user-guide/install/index.html),
you can install ESMValCore by running:

    conda install -c esmvalgroup -c conda-forge esmvalcore

## Installing from source

If you need the very latest features, please see
[CONTRIBUTING.md](https://github.com/ESMValGroup/ESMValCore/blob/master/CONTRIBUTING.md)
for instructions on installing ESMValCore from source.

## Using ESMValCore as a Python library

The ESMValCore package provides various functions for:

-   Finding data in a directory structure typically used for CMIP data

-   Reading CMOR tables and using those to check model and observational data.

-   ESMValTool preprocessor functions based on
    [iris](https://scitools.org.uk/iris/docs/latest/) for e.g. regridding,
    vertical interpolation, statistics, correcting (meta)data errors, extracting
    a time range, etcetera.

## Running ESMValTool

-   Review `config-user.yml`. To customize for your system, create a copy, edit
    and use the command line option `-c` to instruct `esmvaltool` to use your
    custom configuration file.

-   Install the [ESMValTool](https://github.com/ESMValGroup/ESMValTool)
    to run [ESMValTool recipes and diagnostics](https://esmvaltool.readthedocs.io/en/latest/recipes/index.html)

-   Run e.g. `esmvaltool -c ~/config-user.yml examples/recipe_python.yml` after
    downloading the necessary data.

## Getting help

The easiest way to get help if you cannot find the answer in the documentation
on [readthedocs](https://esmvaltool.readthedocs.io), is to open an
[issue on GitHub](https://github.com/ESMValGroup/ESMValCore/issues).

## Contributing

If you would like to contribute a new feature, please have a
look at [CONTRIBUTING.md](https://github.com/ESMValGroup/ESMValCore/blob/master/CONTRIBUTING.md).
