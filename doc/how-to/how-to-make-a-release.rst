.. _how-to-make-a-release:

How to make a release
----------------------

The release manager makes the release, assisted by the release manager of the
previous release, or if that person is not available, another previous release
manager.
Perform the steps listed below with two persons, to reduce the risk of error.

.. note::

   The previous release manager ensures the current release manager has the
   required administrative permissions to make the release.
   Consider the following services:
   `conda-forge <https://github.com/conda-forge/esmvalcore-feedstock>`__,
   `DockerHub <https://hub.docker.com/orgs/esmvalgroup>`__,
   `PyPI <https://pypi.org/project/ESMValCore/>`__, and
   `readthedocs <https://readthedocs.org/dashboard/esmvalcore/users/>`__.

The release of ESMValCore is tied to the release of ESMValTool.
The detailed steps can be found in the ESMValTool
:ref:`documentation <esmvaltool:release_steps>`.
To start the procedure, ESMValCore gets released as a
release candidate to test the recipes in ESMValTool. If bugs are found
during the testing phase of the release candidate, make as many release
candidates for ESMValCore as needed in order to fix them.

.. figure:: figures/release-timeline-doodle-esmvalcore.png
   :target: figures/release-timeline-doodle-esmvalcore.png
   :align:   center

   Example of a release timeline for ESMValCore (in this case for 2.11.0)

To make a new release of the package, be it a release candidate or the final release,
follow these steps:

1. Check that all tests and builds work
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Check that the ``nightly``
  `test run on CircleCI <https://circleci.com/gh/ESMValGroup/ESMValCore/tree/main>`__
  was successful.
- Check that the
  `GitHub Actions test runs <https://github.com/ESMValGroup/ESMValCore/actions>`__
  were successful.
- Check that the documentation builds successfully on
  `readthedocs <https://readthedocs.org/projects/esmvalcore/builds/>`__.
- Check that the
  `Docker images <https://hub.docker.com/repository/docker/esmvalgroup/esmvalcore/builds>`__
  are building successfully.

All tests should pass before making a release (branch).

2. Create a release branch
~~~~~~~~~~~~~~~~~~~~~~~~~~
Create a branch off the ``main`` branch and push it to GitHub.
Ask someone with administrative permissions to set up branch protection rules
for it so only you and the person helping you with the release can push to it.

3. Increase the version number
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The version number is automatically generated from the information provided by
git using [setuptools-scm](https://pypi.org/project/setuptools-scm/), but a
static version number is stored in ``CITATION.cff``.
Make sure to update the version number and release date in ``CITATION.cff``.
See https://semver.org for more information on choosing a version number.
Make a pull request and get it merged into ``main`` and cherry pick it into
the release branch.

4. Add release notes
~~~~~~~~~~~~~~~~~~~~
Use the script
:ref:`esmvaltool/utils/draft_release_notes.py <draft_release_notes.py>`
to create create a draft of the release notes.
This script uses the titles and labels of merged pull requests since the
previous release.
Open a discussion to allow members of the development team to nominate pull
requests as highlights. Add the most voted pull requests as highlights at the
beginning of changelog. After the highlights section, list any backward
incompatible changes that the release may include. The
:ref:`backward compatibility policy<esmvaltool:backward-compatibility-policy>`.
lists the information that should be provided by the developer of any backward
incompatible change. Make sure to also list any deprecations that the release
may include, as well as a brief description on how to upgrade a deprecated feature.
Review the results, and if anything needs changing, change it on GitHub and
re-run the script until the changelog looks acceptable.
Copy the result to the file ``doc/changelog.rst``.
Make a pull request and get it merged into ``main`` and cherry pick it into
the release branch.


5. Make the (pre-)release on GitHub
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Do a final check that all tests on CircleCI and GitHub Actions completed
successfully.
Then click the
`releases tab <https://github.com/ESMValGroup/ESMValCore/releases>`__
and create the new release from the release branch (i.e. not from ``main``).

Create a tag and tick the `This is a pre-release` box if working with a release candidate.

6. Mark the release in the main branch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the (pre-)release is tagged, it is time to merge the release branch back into `main`.
We do this for two reasons, namely, one, to mark the point up to which commits in `main`
have been considered for inclusion into the present release, and, two, to inform
setuptools-scm about the version number so that it creates the correct version number in
`main`.
However, unlike in a normal merge, we do not want to integrate any of the changes from the
release branch into main.
This is because all changes that should be in both branches, i.e. bug fixes, originate from
`main` anyway and the only other changes in the release branch relate to the release itself.
To take this into account, we perform the merge in this case on the command line using `the
ours merge strategy <https://git-scm.com/docs/merge-strategies#Documentation/merge-strategies.txt-ours-1>`__
(``git merge -s ours``), not to be confused with the ``ours`` option to the ort merge strategy
(``git merge -X ours``).
For details about merge strategies, see the above-linked page.
To execute the merge use following sequence of steps

.. code-block:: bash

   git fetch
   git checkout main
   git pull
   git merge -s ours v2.1.x
   git push

Note that the release branch remains intact and you should continue any work on the release
on that branch.

7. Create and upload the PyPI package
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

8. Create the Conda package
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``esmvalcore`` package is published on the `conda-forge conda channel
<https://anaconda.org/conda-forge>`__.
This is done via a pull request on the `esmvalcore-feedstock repository
<https://github.com/conda-forge/esmvalcore-feedstock>`__.

To publish a release candidate, you have to open a pull request yourself.
An example for this can be found `here
<https://github.com/conda-forge/esmvalcore-feedstock/pull/35>`__.
Make sure to use the `rc branch
<https://github.com/conda-forge/esmvalcore-feedstock/tree/rc>`__ as the target
branch for your pull request and follow all instructions given by the linter
bot. The testing of ESMValTool will be performed with the published release candidate.

For the final release, this pull request is automatically opened by a bot.
An example pull request can be found `here
<https://github.com/conda-forge/esmvalcore-feedstock/pull/11>`__.
Follow the instructions by the bot to finalize the pull request.
This step mostly contains updating dependencies that have been changed during
the last release cycle.
Once approved by the `feedstock maintainers
<https://github.com/conda-forge/esmvalcore-feedstock/blob/main/README.md#feedstock-maintainers>`__
they will merge the pull request, which will in turn publish the package on
conda-forge some time later.
Contact the feedstock maintainers if you want to become a maintainer yourself.


9. Check the Docker images
~~~~~~~~~~~~~~~~~~~~~~~~~~

There are two main Docker container images available for ESMValCore on
`Dockerhub <https://hub.docker.com/r/esmvalgroup/esmvalcore/tags>`_:

- ``esmvalgroup/esmvalcore:stable``, built from `docker/Dockerfile <https://github.com/ESMValGroup/ESMValCore/blob/main/docker/Dockerfile>`_,
  this is a tag that is always the same as the latest released version.
  This image is only built by Dockerhub when a new release is created.
- ``esmvalgroup/esmvalcore:development``, built from `docker/Dockerfile.dev <https://github.com/ESMValGroup/ESMValCore/blob/main/docker/Dockerfile.dev>`_,
  this is a tag that always contains the latest conda environment for
  ESMValCore, including any test dependencies.
  It is used by `CircleCI <https://app.circleci.com/pipelines/github/ESMValGroup/ESMValCore>`_ to run the unit tests.
  This speeds up running the tests, as it avoids the need to build the conda
  environment for every test run.
  This image is built by Dockerhub every time there is a new commit to the
  ``main`` branch on Github.

In addition to the two images mentioned above, there is an image available
for every release (e.g. ``esmvalgroup/esmvalcore:v2.5.0``).
When working on the Docker images, always try to follow the
`best practices <https://docs.docker.com/develop/develop-images/dockerfile_best-practices/>`__.

After making the release, check that the Docker image for that release has been
built correctly by

1. checking that the version tag is available on `Dockerhub`_ and the ``stable``
   tag has been updated,
2. running some recipes with the ``stable`` tag Docker container, for example one
   recipe for Python, NCL, and R,
3. running a recipe with a Singularity container built from the ``stable`` tag.

If there is a problem with the automatically built container image, you can fix
the problem and build a new image locally.
For example, to
`build <https://docs.docker.com/engine/reference/commandline/build/>`__ and
`upload <https://docs.docker.com/engine/reference/commandline/push/>`__
the container image for v2.5.0 of the tool run:

.. code-block:: bash

   git checkout v2.5.0
   git clean -x
   docker build -t esmvalgroup/esmvalcore:v2.5.0 . -f docker/Dockerfile
   docker push esmvalgroup/esmvalcore:v2.5.0

(when making updates, you may want to add .post0, .post1, .. to the version
number to avoid overwriting an older tag) and if it is the latest release
that you are updating, also run

.. code-block:: bash

   docker tag esmvalgroup/esmvalcore:v2.5.0 esmvalgroup/esmvalcore:stable
   docker push esmvalgroup/esmvalcore:stable

Note that the ``docker push`` command will overwrite the existing tags on
Dockerhub, but the previous container image will remain available as an
untagged image.
