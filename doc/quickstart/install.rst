.. _install:

Installation
============

Mamba/Conda installation
------------------------

In order to install ESMValCore and its dependencies from
`conda-forge <https://conda-forge.org/>`__, you will first need to install the
`mamba package manager <https://mamba.readthedocs.io>`_. We recommend using
mamba instead of conda because it is faster and offers the same commands.
For a minimal mamba installation (recommended) go to
https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html.

.. tip::

   It is recommended that you always use the latest version of mamba, as
   problems have been reported when trying to use older versions.

Once you have installed mamba, you can install ESMValCore by running:

.. code-block:: bash

    mamba install -c conda-forge esmvalcore

It is also possible to create a new
`Conda environment <https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html#creating-environments>`_
and install ESMValCore into it with a single command:

.. code-block:: bash

    mamba create --name esmvalcore -c conda-forge esmvalcore

Don't forget to activate the newly created environment after the installation:

.. code-block:: bash

    mamba activate esmvalcore

Of course it is also possible to choose a different name than ``esmvalcore`` for the environment.

.. note::

	  Creating a new Conda environment is often much faster and more reliable than trying to update an existing Conda environment.


Pip installation
-----------------

It is also possible to install ESMValCore from `PyPI <https://pypi.org/project/ESMValCore/>`_.
However, this requires first installing dependencies that are not available
on PyPI in some other way.
The list of required dependencies can be found in
:download:`pyproject.toml <../../pyproject.toml>`.

.. warning::

    It is recommended to use the installation with mamba instead, as it may not
    be easy to install the correct versions of all dependencies.

After installing the dependencies that are not available from PyPI_, install
ESMValCore and any remaining Python dependencies with the command:

.. code-block:: bash

    pip install esmvalcore


Docker installation
-----------------------

ESMValCore is also provided through `DockerHub <https://hub.docker.com/u/esmvalgroup/>`_
in the form of docker containers.
See https://docs.docker.com for more information about docker containers and how to
run them.

You can get the latest release with

.. code-block:: bash

   docker pull esmvalgroup/esmvalcore:stable

If you want to use the current main branch, use

.. code-block:: bash

   docker pull esmvalgroup/esmvalcore:latest

To run a container using those images, use:

.. code-block:: bash

   docker run esmvalgroup/esmvalcore:stable --help

Note that the container does not see the data or environmental variables available in the host by default.
You can make data available with ``-v /path:/path/in/container`` and environmental variables with ``-e VARNAME``.

For example, the following command would run a recipe

.. code-block:: bash

   docker run -e HOME -v "$HOME":"$HOME" -v /data:/data esmvalgroup/esmvalcore:stable ~/recipes/recipe_example.yml

with the environmental variable ``$HOME`` available inside the container and the data
in the directories ``$HOME`` and ``/data``, so these can be used to find the configuration, recipe, and data.

It might be useful to define a `bash alias
<https://opensource.com/article/19/7/bash-aliases>`_
or script to abbreviate the above command, for example

.. code-block:: bash

	 alias esmvaltool="docker run -e HOME -v $HOME:$HOME -v /data:/data esmvalgroup/esmvalcore:stable"

would allow using the ``esmvaltool`` command without even noticing that the tool is running inside a Docker container.


Singularity installation
----------------------------

Docker is usually forbidden in clusters due to security reasons. However,
there is a more secure alternative to run containers that is usually available
on them: `Singularity <https://sylabs.io/guides/3.0/user-guide/quick_start.html>`_.

Singularity can use docker containers directly from DockerHub with the
following command

.. code-block:: bash

   singularity run docker://esmvalgroup/esmvalcore:stable ~/recipes/recipe_example.yml

Note that the container does not see the data available in the host by default.
You can make host data available with ``-B /path:/path/in/container``.

It might be useful to define a `bash alias
<https://opensource.com/article/19/7/bash-aliases>`_
or script to abbreviate the above command, for example

.. code-block:: bash

	 alias esmvaltool="singularity run -B $HOME:$HOME -B /data:/data docker://esmvalgroup/esmvalcore:stable"

would allow using the ``esmvaltool`` command without even noticing that the tool is running inside a Singularity container.

Some clusters may not allow to connect to external services, in those cases
you can first create a singularity image locally:

.. code-block:: bash

   singularity build esmvalcore.sif docker://esmvalgroup/esmvalcore:stable

and then upload the image file ``esmvalcore.sif`` to the cluster.
To run the container using the image file ``esmvalcore.sif`` use:

.. code-block:: bash

   singularity run esmvalcore.sif ~/recipes/recipe_example.yml


.. _installation-from-source:

Installation from source
------------------------

.. note::
    If you would like to install the development version of ESMValCore alongside
    ESMValTool, please have a look at
    :ref:`these instructions <esmvaltool:esmvalcore-development-installation>`.

To install from source for development, follow these instructions.

-  `Install pixi <https://pixi.prefix.dev/latest/installation/>`__
-  Clone the ESMValCore Git repository:
   ``git clone https://github.com/ESMValGroup/ESMValCore.git``
-  Go to the source code directory: ``cd ESMValCore``
-  Create the development environment and start a shell in this environment by running: ``pixi shell``
-  Test that your installation was successful by running ``esmvaltool -h``.
-  Install the :ref:`esmvaltool:pre-commit` hooks by running: ``pre-commit install``.

.. tip::

    If you find that solving the environments (i.e. finding out which
    combination of package versions is compatible and can be installed) is
    slow, you can add the ``--frozen`` flag to the commands above to skip the
    solve step. Add ``export PIXI_FROZEN=true`` to your ``~/.bashrc`` file to
    make this the default behavior.

.. tip::

    To exit the pixi environment, run ``exit`` or press ``Ctrl+D``.


Pre-installed versions on HPC clusters / other servers
------------------------------------------------------


If you would like to use pre-installed versions on HPC clusters (currently CEDA-JASMIN and DKRZ-Levante),
and other servers (currently Met Office Linux estate), please have a look at
:ref:`these instructions <esmvaltool:install_on_hpc>`.
