Contributions are very welcome
==============================

We greatly value contributions of any kind.
Contributions could include, but are not limited to documentation improvements, bug reports, new or improved code, scientific and technical code reviews, infrastructure improvements, mailing list and chat participation, community help/building, education and outreach.
We value the time you invest in contributing and strive to make the process as easy as possible.
If you have suggestions for improving the process of contributing, please do not hesitate to propose them.

If you have a bug or other issue to report or just need help, please open an issue on the
`issues tab on the ESMValCore github repository <https://github.com/ESMValGroup/ESMValCore/issues>`__.

If you would like to contribute a new preprocessor function,
:ref:`derived variable <derivation>`, :ref:`fix for a dataset <fixing_data>`, or
another new feature, please discuss your idea with the development team before
getting started, to avoid double work and/or disappointment later.
A good way to do this is to open an
`issue on GitHub <https://github.com/ESMValGroup/ESMValCore/issues>`__.

To get started developing, follow the instructions below.
For help with common new features, please have a look at :doc:`develop/index`.

Getting started
---------------

See :ref:`installation-from-source` for instructions on how to set up a development
installation.

New development should preferably be done in the
`ESMValCore <https://github.com/ESMValGroup/ESMValCore>`__
GitHub repository.
The default git branch is ``master``.
Use this branch to create a new feature branch from and make a pull request
against.
This
`page <https://www.atlassian.com/git/tutorials/comparing-workflows/feature-branch-workflow>`__
offers a good introduction to git branches, but it was written for
BitBucket while we use GitHub, so replace the word BitBucket by GitHub
whenever you read it.

It is recommended that you open a `draft pull
request <https://github.blog/2019-02-14-introducing-draft-pull-requests/>`__
early, as this will cause :ref:`CircleCI to run the unit tests <tests>`,
:ref:`Codacy to analyse your code <code_quality>`, and
:ref:`readthedocs to build the documentation <documentation>`.
It’s also easier to get help from other developers if your code is visible in a
pull request.

.. _pull_request_checklist:

Checklist for pull requests
---------------------------

To clearly communicate up front what is expected from a pull request, we have
the following checklist.
Please try to do everything on the list before requesting a review.
If you are unsure about something on the list, please ask the
`@ESMValGroup/tech-reviewers`_ or `@ESMValGroup/science-reviewers`_ for help
by commenting on your (draft) pull request or by starting a new
`discussion <https://github.com/ESMValGroup/ESMValTool/discussions>`__.

The icons indicate whether the item will be checked during the
:ref:`🛠 Technical review <technical_review>` or
:ref:`🧪 Scientific review <scientific_review>`.

- 🛠 :ref:`The pull request has a descriptive title and labels <descriptive_pr_title>`
- 🛠 Code is written according to the :ref:`code quality guidelines <code_quality>`
- 🧪 and 🛠 Documentation_ is available
- 🧪 The new functionality is scientifically sound and relevant
- 🛠 Unit tests_ have been added
- 🛠 The :ref:`list of authors <authors>` is up to date
- 🛠 Changed dependencies are :ref:`added or removed correctly <dependencies>`
- 🛠 The :ref:`checks shown below the pull request <pull_request_checks>` are successful

If you make backwards incompatible changes to the recipe format:

- Update ESMValTool and link the pull request(s) in the description

.. _descriptive_pr_title:

Descriptive pull request title and label
----------------------------------------

The title of a pull request should clearly describe what the pull request changes.
If you need more text to describe what the pull request does, please add it in
the description.
`Add one or more labels <https://docs.github.com/en/github/managing-your-work-on-github/managing-labels#applying-labels-to-issues-and-pull-requests>`__
to your pull request to indicate the type of change.
At least one of the following
`labels <https://github.com/ESMValGroup/ESMValCore/labels>`__ should be used:
`bug`, `deprecated feature`, `fix for dataset`, `preprocessor`, `cmor`, `api`,
`testing`, `documentation` or `enhancement`.

The titles and labels of pull requests are used to compile the :ref:`changelog`,
therefore it is important that they are easy to understand for people who are
not familiar with the code or people in the project.
Descriptive pull request titles also makes it easier to find back what was
changed when, which is useful in case a bug was introduced.

.. _code_quality:

Code quality
------------

To increase the readability and maintainability or the ESMValCore source
code, we aim to adhere to best practices and coding standards.

We include checks for Python and yaml files, most of which are described in more
detail in the sections below.
This includes checks for invalid syntax and formatting errors.
:ref:`esmvaltool:pre-commit` is a handy tool that can run all of these checks
automatically just before you commit your code.
It knows knows which tool to run for each filetype, and therefore provides
a convenient way to check your code!

Python
~~~~~~

The standard document on best practices for Python code is
`PEP8 <https://www.python.org/dev/peps/pep-0008/>`__ and there is
`PEP257 <https://www.python.org/dev/peps/pep-0257/>`__ for code documentation.
We make use of
`numpy style docstrings <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html>`__
to document Python functions that are visible on
`readthedocs <https://docs.esmvaltool.org>`_.

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

   docformatter -i some_file.py

to run `docformatter <https://github.com/myint/docformatter>`__ which helps formatting the doc strings (such as line length, spaces).

To check if your code adheres to the standard, go to the directory where
the repository is cloned, e.g. ``cd ESMValCore``, and run `prospector <http://prospector.landscape.io/>`_

::

   prospector esmvalcore/preprocessor/_regrid.py

In addition to prospector, we also use `flake8 <https://flake8.pycqa.org/en/latest/>`_
to automatically check for bugs and formatting mistakes.

When you make a pull request, adherence of the Python development best practices
is checked in two ways:

#. As part of the unit tests, flake8_ is run by
   `CircleCI <https://app.circleci.com/pipelines/github/ESMValGroup/ESMValCore>`_,
   see the section on Tests_ for more information.
#. `Codacy <https://app.codacy.com/gh/ESMValGroup/ESMValCore/pullRequests>`_
   is a service that runs prospector (and other code quality tools) on changed
   files and reports the results.
   Click the 'Details' link behind the Codacy check entry and then click
   'View more details on Codacy Production' to see the results of the static
   code analysis done by Codacy_.
   If you need to log in, you can do so using your GitHub account.

The automatic code quality checks by prospector are really helpful to improve
the quality of your code, but they are not flawless.
If you suspect prospector or Codacy may be wrong, please ask the
`@ESMValGroup/tech-reviewers`_ by commenting on your pull request.

Note that running prospector locally will give you quicker and sometimes more
accurate results than waiting for Codacy.

YAML
~~~~

Please use ``yamllint`` to check that your YAML files do not contain mistakes.

Any text file
~~~~~~~~~~~~~

A generic tool to check for common spelling mistakes is
`codespell <https://pypi.org/project/codespell/>`__.

.. _documentation:

Documentation
-------------

What should be documented
~~~~~~~~~~~~~~~~~~~~~~~~~

The documentation lives on `docs.esmvaltool.org <https://docs.esmvaltool.org>`_
and is built using `Sphinx <https://www.sphinx-doc.org>`_.
There are two main ways of adding documentation:

#. As written text in the directory
   `doc <https://github.com/ESMValGroup/ESMValCore/tree/master/doc/>`__.
   When writing
   `reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_
   (``.rst``) files, please try to limit the line length to 80 characters and
   always start a sentence on a new line.
   This makes it easier to review changes to documentation on GitHub.

#. As docstrings or comments in code.
   For Python code, the
   `docstrings <https://www.python.org/dev/peps/pep-0257/>`__
   of Python modules, classes, and functions
   that are mentioned in
   `doc/api <https://github.com/ESMValGroup/ESMValCore/tree/master/doc/api>`__
   are used to generate documentation.
   This results in the :ref:`api`.
   The standard document with best practices on writing docstrings is
   `PEP257 <https://www.python.org/dev/peps/pep-0257/>`__.
   We make use of
   `numpy style docstrings <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html>`__.

Any code documentation that is visible on readthedocs_ should be well
written and adhere to the standards for documentation.
There is no need to write complete numpy style documentation for functions that
are not visible in the :ref`api` chapter on readthedocs.
However, adding a docstring describing what a function does is always a good
idea.
For short functions, a one-line docstring is usually sufficient, but more
complex functions might require slightly more extensive documentation.

For functions that compute scientific results, comments with references to
papers and/or other resources as well as formula numbers should be included.

When making changes to/introducing a new preprocessor function, also update the
:ref:`preprocessor documentation <preprocessor>`.

When reviewing a pull request, always check that documentation is easy to
understand and available in all expected places.

See :ref:`esmvaltool:esmvalcore-documentation-integration` for information on
how the ESMValCore documentation is integrated into the complete ESMValTool
project documentation on readthedocs.

How to build and view the documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Whenever you make a pull request or push new commits to an existing pull
request, readthedocs will automatically build the documentation.
The link to the documentation will be shown in the list of checks below your
pull request, click 'Details' behind the check
``docs/readthedocs.org:esmvaltool`` to preview the documentation.
If all checks were successful, you may need to click 'Show all checks' to see
the individual checks.

To build the documentation on your own computer, go to the directory where the
repository was cloned and run

::

   python setup.py build_sphinx

or

::

   python setup.py build_sphinx -Ea

to build it from scratch.
Make sure that your newly added documentation builds without warnings or
errors and looks correctly formatted.
CircleCI_ will build the documentation with the command

.. code-block:: bash

   python setup.py build_sphinx --warning-is-error

to catch mistakes that can be detected automatically.

The configuration file for Sphinx_ is
`doc/shinx/source/conf.py <https://github.com/ESMValGroup/ESMValTool/blob/master/doc/sphinx/source/conf.py>`_.

When reviewing a pull request, always check that the documentation checks
shown below the pull request were successful.
Successful checks have a green ✓ in front, a ❌ means the test job failed.

.. _tests:

Tests
-----

To check that the code works correctly, there tests available in the
`tests <https://github.com/ESMValGroup/ESMValCore/tree/master/tests>`__
directory.

Contributions to ESMValCore should be covered by unit tests.
New or modified preprocessor functions should preferably also be tested using
the `sample data`_.
If you do not know how to start with writing unit tests, ask the
`@ESMValGroup/tech-reviewers`_ for help by commenting on the pull request and
they will help you get started.

Whenever you make a pull request or push new commits to an existing pull
request, these tests will be run automatically on CircleCI_.
The results appear at the bottom of the pull request.
Click on 'Details' for more information on a specific test job.
To see some of the results on CircleCI, you may need to log in.
You can do so using your GitHub account.

To run the tests on your own computer, go to the directory where the repository
is cloned and run the command

.. code-block:: bash

   pytest

Optionally you can skip tests which require additional dependencies for
supported diagnostic script languages by adding ``-m 'not installation'`` to the
previous command.

Every night, more extensive tests are run to make sure that problems with the
installation of the tool are discovered by the development team before users
encounter them.
These nightly tests have been designed to mimic the installation procedures
described in the documentation, e.g. in the :ref:`install` chapter.
The nightly tests are run using both CircleCI and GitHub Actions, the
result of the tests ran by CircleCI can be seen on the
`CircleCI project page <https://app.circleci.com/pipelines/github/ESMValGroup/ESMValCore?branch=master>`__
and the result of the tests ran by GitHub Actions can be viewed on the
`Actions tab <https://github.com/ESMValGroup/ESMValCore/actions>`__
of the repository.

The configuration of the tests run by CircleCI can be found in the directory
`.circleci <https://github.com/ESMValGroup/ESMValCore/blob/master/.circleci>`__,
while the configuration of the tests run by GitHub Actions can be found in the
directory
`.github/workflows <https://github.com/ESMValGroup/ESMValCore/blob/master/.github/workflows>`__.

When reviewing a pull request, always check that all test jobs on CircleCI_ were
successful.
Successful test jobs have a green ✓ in front, a ❌ means the test job failed.

Sample data
~~~~~~~~~~~

If you need example data to work with, the
`ESMValTool_sample_data <https://github.com/ESMValGroup/ESMValTool_sample_data>`_
repository and package contains samples of CMIP6 data for use with ESMValTool
development, demonstration purposes, and automated testing.
The goal is to keep the repository size small (~ 100 MB), so it can be easily
downloaded and distributed.

The `ESMValTool-sample-data <https://pypi.org/project/ESMValTool-sample-data/>`_
package is installed as part of the developer dependencies and can be used to
test preprocessor functions, see
`tests/sample_data <https://github.com/ESMValGroup/ESMValCore/tree/master/tests/sample_data>`__.

The preprocessing of the sample data can be time-consuming, so some
intermediate results are cached by ``pytest`` to make the tests run faster.
Clear the cache by using running pytest with the ``--cache-clear`` flag.
To avoid running the time consuming tests that use sample data, run
``pytest -m "not use_sample_data"``.
If you are adding new tests using sample data, please mark these as using
sample data by using the
`decorator <https://docs.python.org/3/glossary.html#term-decorator>`__
``@pytest.mark.use_sample_data``.

.. _authors:

List of authors
---------------

If you make a contribution to ESMValCore and you would like to be listed as an
author (e.g. on `Zenodo <https://zenodo.org/record/4525749>`__), please add your
name to the list of authors in ``CITATION.cff`` and generate the entry for the
``.zenodo.json`` file by running the command

::

   pip install cffconvert
   cffconvert --ignore-suspect-keys --outputformat zenodo --outfile .zenodo.json


.. _dependencies:

Adding or removing dependencies
-------------------------------

Before considering adding a new dependency, carefully check that the license of
the dependency you want to add and any of its dependencies are compatible with
the
`Apache 2.0 <https://github.com/ESMValGroup/ESMValCore/blob/master/LICENSE/>`_
license that applies to the ESMValCore.
Note that GPL version 2 license is considered incompatible with the Apache 2.0
license, while the compatibility of GPL version 3 license with the Apache 2.0
license is questionable.
See this `statement <https://www.apache.org/licenses/GPL-compatibility.html>`__
by the authors of the Apache 2.0 license for more information.

The following files contain lists of dependencies

- ``environment.yml``
  contains development dependencies that cannot be installed from
  `PyPI <https://pypi.org/>`__
- ``docs/requirements.txt``
  contains Python dependencies needed to build the documentation that can be
  installed from PyPI
- ``docs/conf.py``
  contains a list of Python dependencies needed to build the documentation that
  cannot be installed from PyPI and need to be mocked when building the
  documentation.
  We do not use conda to build the documentation because this is too time
  consuming.
- ``setup.py``
  contains all Python dependencies, regardless of their installation source
- ``package/meta.yaml``
  contains dependencies for the conda package, all Python and compiled
  dependencies that can be installed from conda should be listed here.

Note that packages may have a different name on
`conda-forge <https://conda-forge.org/>`__ than on PyPI or CRAN.

Several test jobs on CircleCI_ related to the installation of the tool will only
run if you change the dependencies, these will be skipped for most pull
requests.

When reviewing a pull request where dependencies are added or removed, always
check that the changes have been applied in all relevant files.

.. _pull_request_checks:

Pull request checks
-------------------

To check that a pull request is up to standard, several automatic checks are
run when you make a pull request.
Read more about it in the Tests_ and Documentation_ sections.
Successful checks have a green ✓ in front, a ❌ means the check failed.

If you need help with the checks, please ask the technical reviewer of your pull
request for help.
Ask `@ESMValGroup/tech-reviewers`_ if you do not have a technical reviewer yet.

If the checks are broken because of something unrelated to the current
pull request, please check if there is an open issue that reports the problem
and create one if there is no issue yet.
You can attract the attention of the `@ESMValGroup/esmvaltool-coreteam`_ by
mentioning them in the issue if it looks like no-one is working on solving the
problem yet.
The issue needs to be fixed in a separate pull request first.
After that has been merged into the ``master`` branch and all checks are green
again on the ``master`` branch, merge it into your own branch to get the tests
to pass.

When reviewing a pull request, always make sure that all checks were successful.
If the Codacy check keeps failing, please run ``prospector`` locally and if
necessary, ask the pull request author to do the same and to address the
reported issues, see the section on code_quality_ for more information.
Never merge a pull request with failing CircleCI or readthedocs checks.

.. _how-to-make-a-release:

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
:ref:`esmvaltool/utils/draft_release_notes.py <esmvaltool:draft_release_notes.py>`
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


.. _`@ESMValGroup/esmvaltool-coreteam`: https://github.com/orgs/ESMValGroup/teams/esmvaltool-coreteam
.. _`@ESMValGroup/esmvaltool-developmentteam`: https://github.com/orgs/ESMValGroup/teams/esmvaltool-developmentteam
.. _`@ESMValGroup/tech-reviewers`: https://github.com/orgs/ESMValGroup/teams/tech-reviewers
.. _`@ESMValGroup/science-reviewers`: https://github.com/orgs/ESMValGroup/teams/science-reviewers
