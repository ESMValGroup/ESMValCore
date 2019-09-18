# Contributions are very welcome

If you would like to contribute a new diagnostic and recipe or a new feature, please discuss your idea with the development team before getting started, to avoid double work and/or disappointment later. A good way to do this is to open an [issue on GitHub](https://github.com/ESMValGroup/ESMValTool/issues). This is also a good way to get help.

If you have a bug to report, please do so using the [issues tab on the ESMValCore github repository](https://github.com/ESMValGroup/ESMValCore/issues).

To get started developing, follow the instructions below. If you are contributing fixes for a model/dataset, you can find extra instructions [here](https://esmvaltool.readthedocs.io/projects/esmvalcore/en/latest/esmvalcore/fixing_data.html).

## Getting started

To install in development mode, follow these instructions.

-   [Download and install conda](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html) (this should be done even if the system in use already has a preinstalled version of conda, as problems have been reported with NCL when using such a version)
-   To make the `conda` command availble, add `source <prefix>/etc/profile.d/conda.sh` to your `.bashrc` file and restart your shell. If using (t)csh shell, add `source <prefix>/etc/profile.d/conda.csh` to your `.cshrc`/`.tcshrc` file instead.
-   Update conda: `conda update -y conda`
-   Clone the ESMValCore Git repository: `git clone git@github.com:ESMValGroup/ESMValCore`
-   Go to the source code directory: `cd ESMValCore`
-   Create the esmvaltool conda environment `conda env create --name esmvaltool --file environment.yml`
-   Activate the esmvaltool environment: `conda activate esmvaltool`
-   Install in development mode: `pip install -e '.[develop]'`. If you are installing behind a proxy that does not trust the usual pip-urls you can declare them with the option `--trusted-host`, e.g. `pip install --trusted-host=pypi.python.org --trusted-host=pypi.org --trusted-host=files.pythonhosted.org -e .[develop]`
-   Test that your installation was succesful by running `esmvaltool -h`.
-   If you log into a cluster or other device via `ssh` and your origin machine sends the `locale` environment via the `ssh` connection, make sure the environment is set correctly, specifically `LANG` and `LC_ALL` are set correctly (for GB English UTF-8 encoding these variables must be set to `en_GB.UTF-8`; you can set them by adding `export LANG=en_GB.UTF-8` and `export LC_ALL=en_GB.UTF-8` in your origin or login machines' `.profile`)

## Running tests

Go to the directory where the repository is cloned and run `python setup.py test --addopts --installation`. Tests will also be run automatically by [CircleCI](https://circleci.com/gh/ESMValGroup/ESMValCore).

## Code style

To increase the readability and maintainability or the ESMValTool source code, we aim to adhere to best practices and coding standards. All pull requests are reviewed and tested by one or more members of the core development team. For code in all languages, it is highly recommended that you split your code up in functions that are short enough to view without scrolling.

### Python

The standard document on best practices for Python code is [PEP8](https://www.python.org/dev/peps/pep-0008/) and there is [PEP257](https://www.python.org/dev/peps/pep-0257/) for documentation. We make use of [numpy style docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html) to document Python functions that are visible on [readthedocs](https://esmvaltool.readthedocs.io).

Most formatting issues in Python code can be fixed automatically by running the commands

    isort some_file.py

to sort the imports in the standard way and

    yapf -i some_file.py

to add/remove whitespace as required by the standard.

To check if your code adheres to the standard, go to the directory where the repository is cloned, e.g. `cd ESMValTool`.
and run

    prospector esmvaltool/diag_scripts/your_diagnostic/your_script.py

Run

    python setup.py lint

to see the warnings about the code style of the entire project.

We use `pycodestyle` on CircleCI to automatically check that there are no formatting mistakes and Codacy for monitoring (Python) code quality. Running prospector locally will give you quicker and sometimes more accurate results.

### YAML

Please use `yamllint` to check that your YAML files do not contain mistakes.

## Documentation

### What should be documented

Any code documentation that is visible on [readthedocs](https://esmvaltool.readthedocs.io) should be well written and adhere to the standards for documentation for the respective language. Note that there is no need to write extensive documentation for functions that are not visible on readthedocs. However, adding a one line docstring describing what a function does is always a good idea.
When making changes/introducing a new preprocessor function, also update the [preprocessor documentation](https://esmvaltool.readthedocs.io/projects/esmvalcore/en/latest/esmvalcore/preprocessor.html).

### How to build the documentation locally

Go to the directory where the repository is cloned and run

    python setup.py build_sphinx -Ea

Make sure that your newly added documentation builds without warnings or errors.

## Pull requests and code review

It is recommended that you open a pull request early, as this will cause CircleCI to run the unit tests and Codacy to analyse your code. It's also easier to get help from other developers if your code is visible in a pull request.

You can view the results of the automatic checks below your pull request. If one of the tests shows a red cross instead of a green approval sign, please click the link and try to solve the issue. Note that this kind of automated checks make it easier to review code, but they are not flawless, so occasionally Codacy will report false positives.

### Contributing to the ESMValCore package

Contributions to ESMValCore should

-   Preferably be covered by unit tests. Unit tests are mandatory for new preprocessor functions or modifications to existing functions. If you do not know how to start with writing unit tests, let us know in a comment on the pull request and a core development team member will try to help you get started.
-   Be accompanied by appropriate documentation.
-   Introduce no new issues on Codacy.

### List of authors

If you make a (significant) contribution to ESMValCore, please add your name to the list of authors in CITATION.cff and regenerate the file .zenodo.json by running the command

    pip install cffconvert
    cffconvert --ignore-suspect-keys --outputformat zenodo --outfile .zenodo.json
