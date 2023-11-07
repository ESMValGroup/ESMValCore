# ESMValCore package

[![Documentation Status](https://readthedocs.org/projects/esmvalcore/badge/?version=latest)](https://esmvaltool.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3387139.svg)](https://doi.org/10.5281/zenodo.3387139)
[![Chat on Matrix](https://matrix.to/img/matrix-badge.svg)](https://matrix.to/#/#ESMValGroup_Lobby:gitter.im)
[![CircleCI](https://circleci.com/gh/ESMValGroup/ESMValCore/tree/main.svg?style=svg)](https://circleci.com/gh/ESMValGroup/ESMValCore/tree/main)
[![codecov](https://codecov.io/gh/ESMValGroup/ESMValCore/branch/main/graph/badge.svg?token=wQnDzguwq6)](https://codecov.io/gh/ESMValGroup/ESMValCore)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/5d496dea9ef64ec68e448a6df5a65783)](https://www.codacy.com/gh/ESMValGroup/ESMValCore?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=ESMValGroup/ESMValCore&amp;utm_campaign=Badge_Grade)
[![Docker Build Status](https://img.shields.io/docker/cloud/build/esmvalgroup/esmvalcore)](https://hub.docker.com/r/esmvalgroup/esmvalcore/)
[![Anaconda-Server Badge](https://img.shields.io/badge/Anaconda.org-2.9.0-blue.svg)](https://anaconda.org/conda-forge/esmvalcore)
[![Github Actions Test](https://github.com/ESMValGroup/ESMValCore/actions/workflows/run-tests.yml/badge.svg)](https://github.com/ESMValGroup/ESMValCore/actions/workflows/run-tests.yml)

![esmvaltoollogo](https://raw.githubusercontent.com/ESMValGroup/ESMValCore/main/doc/figures/ESMValTool-logo-2.png)

ESMValCore: core functionalities for the ESMValTool, a community diagnostic
and performance metrics tool for routine evaluation of Earth System Models
in the Climate Model Intercomparison Project (CMIP).

# Getting started

Please have a look at the
[documentation](https://docs.esmvaltool.org/projects/esmvalcore/en/latest/quickstart/install.html)
to get started.

## Using the ESMValCore package to run recipes

The ESMValCore package provides the `esmvaltool` command, which can be used to run
[recipes](https://docs.esmvaltool.org/projects/esmvalcore/en/latest/recipe/overview.html)
for working with CMIP-like data.
A large collection of ready to use
[recipes and diagnostics](https://docs.esmvaltool.org/en/latest/recipes/index.html)
is provided by the
[ESMValTool](https://github.com/ESMValGroup/ESMValTool)
package.

## Using ESMValCore as a Python library

The ESMValCore package provides various functions for:

-   Finding data in a directory structure typically used for CMIP data.

-   Reading CMIP/CMOR tables and using those to check model and observational data.

-   ESMValTool preprocessor functions based on
    [iris](https://scitools-iris.readthedocs.io) for e.g. regridding,
    vertical interpolation, statistics, correcting (meta)data errors, extracting
    a time range, etcetera.

read all about it in the
[API documentation](https://docs.esmvaltool.org/projects/esmvalcore/en/latest/api/esmvalcore.html).

## Getting help

The easiest way to get help if you cannot find the answer in the documentation
on [readthedocs](https://docs.esmvaltool.org), is to open an
[issue on GitHub](https://github.com/ESMValGroup/ESMValCore/issues).

## Contributing

Contributions are very welcome, please read our
[contribution guidelines](https://docs.esmvaltool.org/projects/ESMValCore/en/latest/contributing.html)
to get started.
