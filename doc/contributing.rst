Contributions are very welcome
==============================

We greatly value contributions of any kind.
Contributions could include, but are not limited to documentation improvements, bug reports, new or improved code, scientific and technical code reviews, infrastructure improvements, mailing list and chat participation, community help/building, education and outreach.
We value the time you invest in contributing and strive to make the process as easy as possible.
If you have suggestions for improving the process of contributing, please do not hesitate to propose them.

If you have a bug or other issue to report or just need help, please open an issue on the
`issues tab on the ESMValCore github repository <https://github.com/ESMValGroup/ESMValCore/issues>`__.

If you would like to contribute a new preprocessor function, derived variable, fix for a dataset, or another new
feature, please discuss your idea with the development team before
getting started, to avoid double work and/or disappointment later.
A good way to do this is to open an
`issue on GitHub <https://github.com/ESMValGroup/ESMValCore/issues>`__.

To get started developing, follow the instructions below.
For help with common new features, please have a look at :doc:`develop/index`.

Getting started
---------------

To install for development, follow the instructions in :doc:`quickstart/install`.

Running tests
-------------

Go to the directory where the repository is cloned and run
``pytest``. Optionally you can skip tests which require
additional dependencies for supported diagnostic script languages by
adding ``-m 'not installation'`` to the previous command.
Tests will also be run automatically by
`CircleCI <https://circleci.com/gh/ESMValGroup/ESMValCore>`__.

Sample data
-----------

If you need sample data to work with, `this repository <https://github.com/ESMValGroup/ESMValTool_sample_data>`__ contains samples of real data for use with ESMValTool development, demonstration purposes and automated testing. The goal is to keep the repository size small (~ 100 MB), so it can be easily downloaded and distributed.

The data are installed as part of the developer dependencies, and used by some larger tests (i.e. in the `multimodel tests <https://github.com/ESMValGroup/ESMValCore/tree/master/tests/sample_data>`__)

The loading and preprocessing of the data can be somewhat time-consuming (~30 secs) and are cached by ``pytest`` to make the tests more performant.
Clear the cache by using running pytest with the ``--cache-clear`` flag. To avoid running these tests using sample data, use `pytest -m "not use_sample_data"`.
If you are adding new tests using sample data, please use the decorator ``@pytest.mark.use_sample_data``.

Code style
----------

To increase the readability and maintainability or the ESMValCore source
code, we aim to adhere to best practices and coding standards. All pull
requests are reviewed and tested by one or more members of the core
development team. For code in all languages, it is highly recommended
that you split your code up in functions that are short enough to view
without scrolling.

We include checks for Python and yaml files, which are
described in more detail in the sections below.
This includes checks for invalid syntax and formatting errors.
`Pre-commit <https://pre-commit.com/>`__ is a handy tool that can run
all of these checks automatically.
It knows knows which tool to run for each filetype, and therefore provides
a simple way to check your code!


Pre-commit
~~~~~~~~~~

To run ``pre-commit`` on your code, go to the ESMValCore directory
(``cd ESMValCore``) and run

::

   pre-commit run

By default, pre-commit will only run on the files that have been changed,
meaning those that have been staged in git (i.e. after
``git add your_script.py``).

To make it only check some specific files, use

::

   pre-commit run --files your_script.py

or

::

   pre-commit run --files your_script.R

Alternatively, you can configure ``pre-commit`` to run on the staged files before
every commit (i.e. ``git commit``), by installing it as a `git hook <https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks>`__ using

::

   pre-commit install

Pre-commit hooks are used to inspect the code that is about to be committed. The
commit will be aborted if files are changed or if any issues are found that
cannot be fixed automatically. Some issues cannot be fixed (easily), so to
bypass the check, run

::

   git commit --no-verify

or

::

   git commit -n

or uninstall the pre-commit hook

::

   pre-commit uninstall


Python
~~~~~~

The standard document on best practices for Python code is
`PEP8 <https://www.python.org/dev/peps/pep-0008/>`__ and there is
`PEP257 <https://www.python.org/dev/peps/pep-0257/>`__ for
documentation. We make use of `numpy style
docstrings <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html>`__
to document Python functions that are visible on
`readthedocs <https://docs.esmvaltool.org>`__.

Most formatting issues in Python code can be fixed automatically by
running the commands

::

   isort some_file.py

to sort the imports in `the standard way <https://www.python.org/dev/peps/pep-0008/#imports>`__
using `isort <https://pycqa.github.io/isort/>`__ and

::

   yapf -i some_file.py

to add/remove whitespace as required by the standard using `yapf <https://github.com/google/yapf>`__,

::

   docformatter -i your_script.py

to run `docformatter <https://github.com/myint/docformatter>`__ which helps formatting the doc strings (such as line length, spaces).

To check if your code adheres to the standard, go to the directory where
the repository is cloned, e.g. ``cd ESMValCore``, and run `prospector <http://prospector.landscape.io/>`__

::

   prospector esmvaltool/diag_scripts/your_diagnostic/your_script.py

Run

::

   python setup.py lint

to see the warnings about the code style of the entire project.

We use `flake8 <https://flake8.pycqa.org/en/latest/>`__ on CircleCI to automatically check that there are
no formatting mistakes and Codacy for monitoring (Python) code quality.
Running prospector locally will give you quicker and sometimes more
accurate results.

YAML
~~~~

Please use ``yamllint`` to check that your YAML files do not contain
mistakes.

Any text file
~~~~~~~~~~~~~

A generic tool to check for common spelling mistakes is
`codespell <https://pypi.org/project/codespell/>`__.

Documentation
-------------

What should be documented
~~~~~~~~~~~~~~~~~~~~~~~~~

Any code documentation that is visible on
`readthedocs <https://docs.esmvaltool.org>`__ should be well
written and adhere to the standards for documentation for the respective
language. Note that there is no need to write extensive documentation
for functions that are not visible on readthedocs. However, adding a one
line docstring describing what a function does is always a good idea.
When making changes/introducing a new preprocessor function, also update
the `preprocessor
documentation <https://docs.esmvaltool.org/projects/ESMValCore/en/latest/recipe/preprocessor.html>`__.

How to build the documentation locally
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Go to the directory where the repository is cloned and run

::

   python setup.py build_sphinx -Ea

Make sure that your newly added documentation builds without warnings or
errors.

Branches, pull requests and code review
---------------------------------------

The default git branch is ``master``. Use this branch to create a new
feature branch from and make a pull request against. This
`page <https://www.atlassian.com/git/tutorials/comparing-workflows/feature-branch-workflow>`__
offers a good introduction to git branches, but it was written for
BitBucket while we use GitHub, so replace the word BitBucket by GitHub
whenever you read it.

It is recommended that you open a `draft pull
request <https://github.blog/2019-02-14-introducing-draft-pull-requests/>`__
early, as this will cause CircleCI to run the unit tests and Codacy to
analyse your code. It’s also easier to get help from other developers if
your code is visible in a pull request.

You also must assign at least one `label <https://docs.github.com/en/github/managing-your-work-on-github/managing-labels#applying-labels-to-issues-and-pull-requests>`__
to it as they are used to organize the changelog. At least one of the following
ones must be used: `bug`, `deprecated feature`, `fix for dataset`,
`preprocessor`, `cmor`, `api`, `testing`, `documentation` or `enhancement`.

You can view the results of the automatic checks below your pull
request. If one of the tests shows a red cross instead of a green
approval sign, please click the link and try to solve the issue. Note
that this kind of automated checks make it easier to review code, but
they are not flawless, so occasionally Codacy will report false
positives.

Contributing to the ESMValCore package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Contributions to ESMValCore should

-  Preferably be covered by unit tests. Unit tests are mandatory for new
   preprocessor functions or modifications to existing functions. If you
   do not know how to start with writing unit tests, let us know in a
   comment on the pull request and a core development team member will
   try to help you get started.
-  Be accompanied by appropriate documentation.
-  Introduce no new issues on Codacy.

List of authors
~~~~~~~~~~~~~~~

If you make a contribution to ESMValCore and would like to be listed as an
author, please add your name to the list of authors in CITATION.cff and
regenerate the file .zenodo.json by running the command

::

   pip install cffconvert
   cffconvert --ignore-suspect-keys --outputformat zenodo --outfile .zenodo.json

How to make a release
---------------------

The release manager makes the release, assisted by the release manager of the
previous release, or if that person is not available, another previous release
manager. Perform the steps listed below with two persons, to reduce the risk of
error.

To make a new release of the package, follow these steps:

1. Check the tests on GitHub Actions and CircleCI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Check the ``nightly``
`build on CircleCI <https://circleci.com/gh/ESMValGroup/ESMValCore/tree/master>`__
and the
`GitHub Actions run <https://github.com/ESMValGroup/ESMValCore/actions>`__.
All tests should pass before making a release (branch).

2. Create a release branch
~~~~~~~~~~~~~~~~~~~~~~~~~~
Create a branch off the ``master`` branch and push it to GitHub.
Ask someone with administrative permissions to set up branch protection rules
for it so only you and the person helping you with the release can push to it.
Announce the name of the branch in an issue and ask the members of the
`ESMValTool development team <https://github.com/orgs/ESMValGroup/teams/esmvaltool-developmentteam>`__
to run their favourite recipe using this branch.

3. Increase the version number
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The version number is stored in ``esmvalcore/_version.py``,
``package/meta.yaml``, ``CITATION.cff``. Make sure to update all files.
Also update the release date in ``CITATION.cff``.
See https://semver.org for more information on choosing a version number.
Make a pull request and get it merged into ``master`` and cherry pick it into
the release branch.

4. Add release notes
~~~~~~~~~~~~~~~~~~~~
Use the script
`esmvaltool/utils/draft_release_notes.py <https://docs.esmvaltool.org/en/latest/utils.html#draft-release-notes-py>`__
to create create a draft of the release notes.
This script uses the titles and labels of merged pull requests since the
previous release.
Review the results, and if anything needs changing, change it on GitHub and
re-run the script until the changelog looks acceptable.
Copy the result to the file ``doc/changelog.rst``.
Make a pull request and get it merged into ``master`` and cherry pick it into
the release branch..

5. Cherry pick bugfixes into the release branch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If a bug is found and fixed (i.e. pull request merged into the
``master`` branch) during the period of testing, use the command
``git cherry-pick`` to include the commit for this bugfix into
the release branch.
When the testing period is over, make a pull request to update
the release notes with the latest changes, get it merged into
``master`` and cherry-pick it into the release branch.

6. Make the release on GitHub
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Do a final check that all tests on CircleCI and GitHub Actions completed
successfully.
Then click the
`releases tab <https://github.com/ESMValGroup/ESMValCore/releases>`__
and create the new release from the release branch (i.e. not from ``master``).

7. Create and upload the Conda package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The package is automatically uploaded to the
`ESMValGroup conda channel <https://anaconda.org/esmvalgroup/esmvalcore>`__
by a GitHub action.
If this has failed for some reason, build and upload the package manually by
following the instructions below.

Follow these steps to create a new conda package:

-  Check out the tag corresponding to the release,
   e.g. ``git checkout tags/v2.1.0``
-  Make sure your current working directory is clean by checking the output
   of ``git status`` and by running ``git clean -xdf`` to remove any files
   ignored by git.
-  Edit ``package/meta.yaml`` and uncomment the lines starting with ``git_rev`` and
   ``git_url``, remove the line starting with ``path`` in the ``source``
   section.
-  Activate the base environment ``conda activate base``
-  Install the required packages:
   ``conda install -y conda-build conda-verify ripgrep anaconda-client``
-  Run ``conda build package -c conda-forge -c esmvalgroup`` to build the
   conda package
-  If the build was successful, upload the package to the esmvalgroup
   conda channel, e.g.
   ``anaconda upload --user esmvalgroup /path/to/conda/conda-bld/noarch/esmvalcore-2.2.0-py_0.tar.bz2``.

8. Create and upload the PyPI package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The package is automatically uploaded to the
`PyPI <https://pypi.org/project/ESMValCore/>`__
by a GitHub action.
If has failed for some reason, build and upload the package manually by
following the instructions below.

Follow these steps to create a new Python package:

-  Check out the tag corresponding to the release,
   e.g. ``git checkout tags/v2.1.0``
-  Make sure your current working directory is clean by checking the output
   of ``git status`` and by running ``git clean -xdf`` to remove any files
   ignored by git.
-  Install the required packages:
   ``python3 -m pip install --upgrade pep517 twine``
-  Build the package:
   ``python3 -m pep517.build --source --binary --out-dir dist/ .``
   This command should generate two files in the ``dist`` directory, e.g.
   ``ESMValCore-2.2.0-py3-none-any.whl`` and ``ESMValCore-2.2.0.tar.gz``.
-  Upload the package:
   ``python3 -m twine upload dist/*``
   You will be prompted for an API token if you have not set this up
   before, see
   `here <https://pypi.org/help/#apitoken>`__ for more information.

You can read more about this in
`Packaging Python Projects <https://packaging.python.org/tutorials/packaging-projects/>`__.
