.. _contributing:

Contributions are very welcome
==============================

We greatly value contributions of any kind.
Contributions could include, but are not limited to documentation improvements, bug reports, new or improved code, scientific and technical code reviews, infrastructure improvements, mailing list and chat participation, community help/building, education and outreach.
We value the time you invest in contributing and strive to make the process as easy as possible.
If you have suggestions for improving the process of contributing, please do not hesitate to propose them.

If you have a bug or other issue to report or just need help, please open an issue on the
`issues tab on the ESMValCore github repository <https://github.com/ESMValGroup/ESMValCore/issues>`__.

If you would like to contribute a new
:ref:`preprocessor function <preprocessor_function>`,
:ref:`derived variable <derivation>`, :ref:`fix for a dataset <fixing_data>`, or
another new feature, please discuss your idea with the development team before
getting started, to avoid double work and/or disappointment later.
A good way to do this is to open an
`issue <https://github.com/ESMValGroup/ESMValCore/issues>`_ on GitHub.

Getting started
---------------

See :ref:`installation-from-source` for instructions on how to set up a development
installation.

New development should preferably be done in the
`ESMValCore <https://github.com/ESMValGroup/ESMValCore>`__
GitHub repository.
The default git branch is ``main``.
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

:ref:`Make small pull requests <easy_review>`, the ideal pull requests changes
just a few files and adds/changes no more than 100 lines of production code.
The amount of test code added can be more extensive, but changes to existing
test code should be made sparingly.

Design considerations
~~~~~~~~~~~~~~~~~~~~~

When making changes, try to respect the current structure of the program.
If you need to make major changes to the structure of program to add a feature,
chances are that you have either not come up with the
most optimal design or the feature is not a very good fit for the tool.
Discuss your feature with the `@ESMValGroup/esmvaltool-coreteam`_ in an issue_
to find a solution.

Please keep the following considerations in mind when programming:

- Changes should preferably be :ref:`backward compatible <backward_compatibility>`.
- Apply changes gradually and change no more than a few files in a single pull
  request, but do make sure every pull request in itself brings a meaningful
  improvement.
  This reduces the risk of breaking existing functionality and making
  :ref:`backward incompatible <backward_compatibility>` changes, because it
  helps you as well as the reviewers of your pull request to better understand
  what exactly is being changed.
- :ref:`preprocessor_functions` are Python functions (and not classes) so they
  are easy to understand and implement for scientific contributors.
- No additional CMOR checks should be implemented inside preprocessor functions.
  The input cube is fixed and confirmed to follow the specification in
  `esmvalcore/cmor/tables <https://github.com/ESMValGroup/ESMValCore/tree/main/esmvalcore/cmor/tables>`__
  before applying any other preprocessor functions.
  This design helps to keep the preprocessor functions and diagnostics scripts
  that use the preprocessed data from the tool simple and reliable.
  See :ref:`cmor_table_configuration` for the mapping from ``project`` in the
  recipe to the relevant CMOR table.
- The ESMValCore package is based on :ref:`iris <iris_docs>`.
  Preprocessor functions should preferably be small and just call the relevant
  iris code.
  Code that is more involved and more broadly applicable than just in the
  ESMValCore, should be implemented in iris instead.
- Any settings in the recipe that can be checked before loading the data should
  be checked at the :ref:`task creation stage <Diagnostics>`.
  This avoids that users run a recipe for several hours before finding out they
  made a mistake in the recipe.
  No data should be processed or files written while creating the tasks.
- CMOR checks should provide a good balance between reliability of the tool
  and ease of use.
  Several :ref:`levels of strictness of the checks <cmor_check_strictness>`
  are available to facilitate this.
- Keep your code short and simple: we would like to make contributing as easy as
  possible.
  For example, avoid implementing complicated class inheritance structures and
  `boilerplate <https://stackoverflow.com/questions/3992199/what-is-boilerplate-code>`__
  code.
- If you find yourself copy-pasting a piece of code and making minor changes
  to every copy, instead put the repeated bit of code in a function that you can
  re-use, and provide the changed bits as function arguments.
- Be careful when changing existing unit tests to make your new feature work.
  You might be breaking existing features if you have to change existing tests.

Finally, if you would like to improve the design of the tool, discuss your plans
with the `@ESMValGroup/esmvaltool-coreteam`_ to make sure you understand the
current functionality and you all agree on the new design.

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

In the ESMValTool community we use
:ref:`pull request reviews <esmvaltool:reviewing>` to ensure all code and
documentation contributions are of good quality.
The icons indicate whether the item will be checked during the
:ref:`🛠 Technical review <technical_review>` or
:ref:`🧪 Scientific review <scientific_review>`.

- 🧪 The new functionality is :ref:`relevant and scientifically sound<scientific_relevance>`
- 🛠 :ref:`The pull request has a descriptive title and labels <descriptive_pr_title>`
- 🛠 Code is written according to the :ref:`code quality guidelines <code_quality>`
- 🧪 and 🛠 Documentation_ is available
- 🛠 Unit tests_ have been added
- 🛠 Changes are :ref:`backward compatible <backward_compatibility>`
- 🛠 Changed :ref:`dependencies have been added or removed correctly <dependencies>`
- 🛠 The :ref:`list of authors <authors>` is up to date
- 🛠 The :ref:`checks shown below the pull request <pull_request_checks>` are successful

.. _scientific_relevance:

Scientific relevance
--------------------

The proposed changes should be relevant for the larger scientific community.
The implementation of new features should be scientifically sound; e.g.
the formulas used in new preprocesssor functions should be accompanied by the
relevant references and checked for correctness by the scientific reviewer.
The `CF Conventions <https://cfconventions.org/>`_ as well as additional
standards imposed by `CMIP <https://www.wcrp-climate.org/wgcm-cmip>`_ should be
followed whenever possible.

.. _descriptive_pr_title:

Pull request title and label
----------------------------

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
a convenient way to check your code.

Python
~~~~~~

The standard document on best practices for Python code is
`PEP8 <https://www.python.org/dev/peps/pep-0008/>`__ and there is
`PEP257 <https://www.python.org/dev/peps/pep-0257/>`__ for code documentation.
We make use of
`numpy style docstrings <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html>`__
to document Python functions that are visible on
`readthedocs <https://docs.esmvaltool.org>`_.

To check if your code adheres to the standard, go to the directory where
the repository is cloned, e.g. ``cd ESMValCore``, and run `prospector <http://prospector.landscape.io/>`_

::

   prospector esmvalcore/preprocessor/_regrid.py

In addition to prospector, we use `flake8 <https://flake8.pycqa.org/en/latest/>`_
to automatically check for bugs and formatting mistakes and
`mypy <https://mypy.readthedocs.io>`_ for checking that
`type hints <https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html>`_ are
correct.
Note that `type hints`_ are completely optional, but if you do choose to add
them, they should be correct.

When you make a pull request, adherence to the Python development best practices
is checked in two ways:

#. As part of the unit tests, flake8_ and mypy_ are run by
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

to run `docformatter <https://github.com/myint/docformatter>`__ which helps
formatting the docstrings (such as line length, spaces).

YAML
~~~~

Please use `yamllint <https://yamllint.readthedocs.io>`_ to check that your
YAML files do not contain mistakes.
``yamllint`` checks for valid syntax, common mistakes like key repetition and
cosmetic problems such as line length, trailing spaces, wrong indentation, etc.

Any text file
~~~~~~~~~~~~~

A generic tool to check for common spelling mistakes is
`codespell <https://pypi.org/project/codespell/>`__.

.. _documentation:

Documentation
-------------

The documentation lives on `docs.esmvaltool.org <https://docs.esmvaltool.org>`_.

Adding documentation
~~~~~~~~~~~~~~~~~~~~

The documentation is built by readthedocs_ using `Sphinx <https://www.sphinx-doc.org>`_.
There are two main ways of adding documentation:

#. As written text in the directory
   `doc <https://github.com/ESMValGroup/ESMValCore/tree/main/doc/>`__.
   When writing
   `reStructuredText <https://www.sphinx-doc.org/en/main/usage/restructuredtext/basics.html>`_
   (``.rst``) files, please try to limit the line length to 80 characters and
   always start a sentence on a new line.
   This makes it easier to review changes to documentation on GitHub.

#. As docstrings or comments in code.
   For Python code, only the
   `docstrings <https://www.python.org/dev/peps/pep-0257/>`__
   of Python modules, classes, and functions
   that are mentioned in
   `doc/api <https://github.com/ESMValGroup/ESMValCore/tree/main/doc/api>`__
   are used to generate the online documentation.
   This results in the :ref:`api`.
   The standard document with best practices on writing docstrings is
   `PEP257 <https://www.python.org/dev/peps/pep-0257/>`__.
   For the API documentation, we make use of
   `numpy style docstrings <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html>`__.

What should be documented
~~~~~~~~~~~~~~~~~~~~~~~~~

Functionality that is visible to users should be documented.
Any documentation that is visible on readthedocs_ should be well
written and adhere to the standards for documentation.
Examples of this include:

- The :ref:`recipe <recipe_overview>`
- Preprocessor :ref:`functions <preprocessor_functions>` and their
  :ref:`use from the recipe <preprocessor>`
- :ref:`Configuration options <config>`
- :ref:`Installation <install>`
- :ref:`Output files <outputdata>`
- :ref:`Command line interface <running>`
- :ref:`Diagnostic script interfaces <interfaces>`
- :ref:`The experimental Python interface <experimental_api>`

Note that:

- For functions that compute scientific results, comments with references to
  papers and/or other resources as well as formula numbers should be included.
- When making changes to/introducing a new preprocessor function, also update the
  :ref:`preprocessor documentation <preprocessor>`.
- There is no need to write complete numpy style documentation for functions that
  are not visible in the :ref:`api` chapter on readthedocs.
  However, adding a docstring describing what a function does is always a good
  idea.
  For short functions, a one-line docstring is usually sufficient, but more
  complex functions might require slightly more extensive documentation.

When reviewing a pull request, always check that documentation is easy to
understand and available in all expected places.

How to build and view the documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Whenever you make a pull request or push new commits to an existing pull
request, readthedocs will automatically build the documentation.
The link to the documentation will be shown in the list of checks below your
pull request.
Click 'Details' behind the check ``docs/readthedocs.org:esmvaltool`` to preview
the documentation.
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
CircleCI_ will build the documentation with the command:

.. code-block:: bash

   python setup.py build_sphinx --warning-is-error

This will catch mistakes that can be detected automatically.

The configuration file for Sphinx_ is
`doc/shinx/source/conf.py <https://github.com/ESMValGroup/ESMValTool/blob/main/doc/sphinx/source/conf.py>`_.

See :ref:`esmvaltool:esmvalcore-documentation-integration` for information on
how the ESMValCore documentation is integrated into the complete ESMValTool
project documentation on readthedocs.

When reviewing a pull request, always check that the documentation checks
shown below the pull request were successful.

.. _tests:

Tests
-----

To check that the code works correctly, there tests available in the
`tests <https://github.com/ESMValGroup/ESMValCore/tree/main/tests>`__ directory.
We use `pytest <https://docs.pytest.org>`_ to write and run our tests.

Contributions to ESMValCore should be
`covered by unit tests <https://the-turing-way.netlify.app/reproducible-research/testing/testing-guidance.html#aim-to-have-a-good-code-coverage>`_.
Have a look at the existing tests in the ``tests`` directory for inspiration on
how to write your own tests.
If you do not know how to start with writing unit tests, ask the
`@ESMValGroup/tech-reviewers`_ for help by commenting on the pull request and
they will try to help you.
It is also recommended that you have a look at the pytest_ documentation at some
point when you start writing your own tests.

Running tests
~~~~~~~~~~~~~

To run the tests on your own computer, go to the directory where the repository
is cloned and run the command

.. code-block:: bash

   pytest

Optionally you can skip tests which require additional dependencies for
supported diagnostic script languages by adding ``-m 'not installation'`` to the
previous command. To only run tests from a single file, run the command

.. code-block:: bash

   pytest tests/unit/test_some_file.py

If you would like to avoid loading the default pytest configuration from
`setup.cfg <https://github.com/ESMValGroup/ESMValCore/blob/main/setup.cfg>`_
because this can be a bit slow for running just a few tests, use

.. code-block:: bash

   pytest -c /dev/null tests/unit/test_some_file.py

Use

.. code-block:: bash

    pytest --help

for more information on the available commands.

Whenever you make a pull request or push new commits to an existing pull
request, the tests in the ``tests`` directory of the branch associated with the
pull request will be run automatically on CircleCI_.
The results appear at the bottom of the pull request.
Click on 'Details' for more information on a specific test job.

When reviewing a pull request, always check that all test jobs on CircleCI_ were
successful.

Test coverage
~~~~~~~~~~~~~

To check which parts of your code are `covered by unit tests`_, open the file
``test-reports/coverage_html/index.html`` (available after running a ``pytest``
command) and browse to the relevant file.

CircleCI will upload the coverage results from running the tests to codecov and
Codacy.
`codecov <https://app.codecov.io/gh/ESMValGroup/ESMValCore/pulls>`_ is a service
that will comment on pull requests with a summary of the test coverage.
If codecov_ reports that the coverage has decreased, check the report and add
additional tests.
Alternatively, it is also possible to view code coverage on Codacy_ (click the
Files tab) and CircleCI_ (open the ``tests`` job and click the ARTIFACTS tab).
To see some of the results on CircleCI, Codacy, or codecov, you may need to log
in; you can do so using your GitHub account.

When reviewing a pull request, always check that new code is covered by unit
tests and codecov_ reports an increased coverage.

.. _sample_data_tests:

Sample data
~~~~~~~~~~~

New or modified preprocessor functions should preferably also be tested using
the sample data.
These tests are located in
`tests/sample_data <https://github.com/ESMValGroup/ESMValCore/tree/main/tests/sample_data>`__.
Please mark new tests that use the sample data with the
`decorator <https://docs.python.org/3/glossary.html#term-decorator>`__
``@pytest.mark.use_sample_data``.

The `ESMValTool_sample_data <https://github.com/ESMValGroup/ESMValTool_sample_data>`_
repository contains samples of CMIP6 data for testing ESMValCore.
The `ESMValTool-sample-data <https://pypi.org/project/ESMValTool-sample-data/>`_
package is installed as part of the developer dependencies.
The size of the package is relatively small (~ 100 MB), so it can be easily
downloaded and distributed.

Preprocessing the sample data can be time-consuming, so some
intermediate results are cached by pytest to make the tests run faster.
If you suspect the tests are failing because the cache is invalid, clear it by
running

.. code-block:: bash

   pytest --cache-clear

To avoid running the time consuming tests that use sample data altogether, run

.. code-block:: bash

   pytest -m "not use_sample_data"


Automated testing
~~~~~~~~~~~~~~~~~

Whenever you make a pull request or push new commits to an existing pull
request, the tests in the ``tests`` of the branch associated with the
pull request will be run automatically on CircleCI_.

Every night, more extensive tests are run to make sure that problems with the
installation of the tool are discovered by the development team before users
encounter them.
These nightly tests have been designed to follow the installation procedures
described in the documentation, e.g. in the :ref:`install` chapter.
The nightly tests are run using both CircleCI and GitHub Actions.
The result of the tests ran by CircleCI can be seen on the
`CircleCI project page <https://app.circleci.com/pipelines/github/ESMValGroup/ESMValCore?branch=main>`__
and the result of the tests ran by GitHub Actions can be viewed on the
`Actions tab <https://github.com/ESMValGroup/ESMValCore/actions>`__
of the repository.

The configuration of the tests run by CircleCI can be found in the directory
`.circleci <https://github.com/ESMValGroup/ESMValCore/blob/main/.circleci>`__,
while the configuration of the tests run by GitHub Actions can be found in the
directory
`.github/workflows <https://github.com/ESMValGroup/ESMValCore/blob/main/.github/workflows>`__.

.. _backward_compatibility:

Backward compatibility
----------------------

The ESMValCore package is used by many people to run their recipes.
Many of these recipes are maintained in the public
`ESMValTool <https://github.com/ESMValGroup/ESMValTool>`_ repository, but
there are also users who choose not to share their work there.
While our commitment is first and foremost to users who do share their recipes
in the ESMValTool repository, we still try to be nice to all of the ESMValCore
users.
When making changes, e.g. to the :ref:`recipe format <recipe_overview>`, the
:ref:`diagnostic script interface <interfaces>`, the public
:ref:`Python API <api>`, or the :ref:`configuration file format <config>`,
keep in mind that this may affect many users.
To keep the tool user friendly, try to avoid making changes that are not
backward compatible, i.e. changes that require users to change their existing
recipes, diagnostics, configuration files, or scripts.

If you really must change the public interfaces of the tool, always discuss this
with the `@ESMValGroup/esmvaltool-coreteam`_.
Try to deprecate the feature first by issuing a :py:class:`DeprecationWarning`
using the :py:mod:`warnings` module and schedule it for removal three
`minor versions <https://semver.org/>`__ from the latest released version.
For example, when you deprecate a feature in a pull request that will be
included in version 2.3, that feature could be removed in version 2.5.
Mention the version in which the feature will be removed in the deprecation
message.
Label the pull request with the
`deprecated feature <https://github.com/ESMValGroup/ESMValCore/labels/deprecated%20feature>`__
label.
When deprecating a feature, please follow up by actually removing the feature
in due course.

If you must make backward incompatible changes, you need to update the available
recipes in ESMValTool and link the ESMValTool pull request(s) in the ESMValCore
pull request description.
You can ask the `@ESMValGroup/esmvaltool-recipe-maintainers`_ for help with
updating existing recipes, but please be considerate of their time.

When reviewing a pull request, always check for backward incompatible changes
and make sure they are needed and have been discussed with the
`@ESMValGroup/esmvaltool-coreteam`_.
Also, make sure the author of the pull request has created the accompanying pull
request(s) to update the ESMValTool, before merging the ESMValCore pull request.

.. _dependencies:

Dependencies
------------

Before considering adding a new dependency, carefully check that the
`license <https://the-turing-way.netlify.app/reproducible-research/licensing/licensing-software.html>`__
of the dependency you want to add and any of its dependencies are
`compatible <https://the-turing-way.netlify.app/reproducible-research/licensing/licensing-compatibility.html>`__
with the
`Apache 2.0 <https://github.com/ESMValGroup/ESMValCore/blob/main/LICENSE/>`_
license that applies to the ESMValCore.
Note that GPL version 2 license is considered incompatible with the Apache 2.0
license, while the compatibility of GPL version 3 license with the Apache 2.0
license is questionable.
See this `statement <https://www.apache.org/licenses/GPL-compatibility.html>`__
by the authors of the Apache 2.0 license for more information.

When adding or removing dependencies, please consider applying the changes in
the following files:

- ``environment.yml``
  contains development dependencies that cannot be installed from
  `PyPI <https://pypi.org/>`_
- ``docs/requirements.txt``
  contains Python dependencies needed to build the documentation that can be
  installed from PyPI
- ``docs/conf.py``
  contains a list of Python dependencies needed to build the documentation that
  cannot be installed from PyPI and need to be mocked when building the
  documentation.
  (We do not use conda to build the documentation because this is too time
  consuming.)
- ``setup.py``
  contains all Python dependencies, regardless of their installation source
- ``package/meta.yaml``
  contains dependencies for the conda package; all Python and compiled
  dependencies that can be installed from conda should be listed here

Note that packages may have a different name on
`conda-forge <https://conda-forge.org/>`__ than on PyPI_.

Several test jobs on CircleCI_ related to the installation of the tool will only
run if you change the dependencies.
These will be skipped for most pull requests.

When reviewing a pull request where dependencies are added or removed, always
check that the changes have been applied in all relevant files.

.. _authors:

List of authors
---------------

If you make a contribution to ESMValCore and you would like to be listed as an
author (e.g. on `Zenodo <https://zenodo.org/record/4525749>`__), please add your
name to the list of authors in ``CITATION.cff`` and generate the entry for the
``.zenodo.json`` file by running the commands

::

   pip install cffconvert
   cffconvert --ignore-suspect-keys --outputformat zenodo --outfile .zenodo.json

Presently, this method unfortunately discards entries `communities`
and `grants` from that file; please restore them manually, or
alternately proceed with the addition manually

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
pull request, please check if there is an open issue that reports the problem.
Create one if there is no issue yet.
You can attract the attention of the `@ESMValGroup/esmvaltool-coreteam`_ by
mentioning them in the issue if it looks like no-one is working on solving the
problem yet.
The issue needs to be fixed in a separate pull request first.
After that has been merged into the ``main`` branch and all checks on this
branch are green again, merge it into your own branch to get the tests to pass.

When reviewing a pull request, always make sure that all checks were successful.
If the Codacy check keeps failing, please run prospector locally.
If necessary, ask the pull request author to do the same and to address the
reported issues.
See the section on code_quality_ for more information.
Never merge a pull request with failing CircleCI or readthedocs checks.


.. _how-to-make-a-release:

Making a release
----------------

The release manager makes the release, assisted by the release manager of the
previous release, or if that person is not available, another previous release
manager. Perform the steps listed below with two persons, to reduce the risk of
error.

To make a new release of the package, follow these steps:

1. Check the tests on GitHub Actions and CircleCI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Check the ``nightly``
`build on CircleCI <https://circleci.com/gh/ESMValGroup/ESMValCore/tree/main>`__
and the
`GitHub Actions run <https://github.com/ESMValGroup/ESMValCore/actions>`__.
All tests should pass before making a release (branch).

2. Create a release branch
~~~~~~~~~~~~~~~~~~~~~~~~~~
Create a branch off the ``main`` branch and push it to GitHub.
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
Make a pull request and get it merged into ``main`` and cherry pick it into
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
Make a pull request and get it merged into ``main`` and cherry pick it into
the release branch..

5. Cherry pick bugfixes into the release branch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If a bug is found and fixed (i.e. pull request merged into the
``main`` branch) during the period of testing, use the command
``git cherry-pick`` to include the commit for this bugfix into
the release branch.
When the testing period is over, make a pull request to update
the release notes with the latest changes, get it merged into
``main`` and cherry-pick it into the release branch.

6. Make the release on GitHub
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Do a final check that all tests on CircleCI and GitHub Actions completed
successfully.
Then click the
`releases tab <https://github.com/ESMValGroup/ESMValCore/releases>`__
and create the new release from the release branch (i.e. not from ``main``).

7. Create and upload the Conda package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The package is automatically uploaded to the
`ESMValGroup conda channel <https://anaconda.org/esmvalgroup/esmvalcore>`__
by a GitHub action (note that this is an obsolete procedure for the main package upload,
since the main package is now uploaded to
`conda-forge conda channel <https://anaconda.org/conda-forge>`__ via
the upload to PyPi, but we still upload to the esmvalgroup channel as a backup option;
also the upload to esmvalcore gives us a chance to verify it immediately after upload).
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
-  Run ``conda build package -c conda-forge`` to build the
   conda package
-  If the build was successful, upload the package to the esmvalgroup
   conda channel, e.g.
   ``anaconda upload --user esmvalgroup /path/to/conda/conda-bld/noarch/esmvalcore-2.3.1-py_0.tar.bz2``.

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
   ``ESMValCore-2.3.1-py3-none-any.whl`` and ``ESMValCore-2.3.1.tar.gz``.
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
.. _`@ESMValGroup/esmvaltool-recipe-maintainers`: https://github.com/orgs/ESMValGroup/teams/esmvaltool-recipe-maintainers
