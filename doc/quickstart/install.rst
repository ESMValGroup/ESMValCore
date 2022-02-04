.. _install:

Installation
============

Conda installation
------------------

In order to install the Conda package, you will need to install `Conda <https://docs.conda.io>`_ first.
For a minimal conda installation (recommended) go to https://conda.io/miniconda.html.
It is recommended that you always use the latest version of conda, as problems have been reported when trying to use older versions.

Once you have installed conda, you can install ESMValCore by running:

.. code-block:: bash

    conda install -c conda-forge esmvalcore

It is also possible to create a new
`Conda environment <https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html#managing-environments>`_
and install ESMValCore into it with a single command:

.. code-block:: bash

    conda create --name esmvalcore -c conda-forge esmvalcore 'python=3.10'

Don't forget to activate the newly created environment after the installation:

.. code-block:: bash

    conda activate esmvalcore

Of course it is also possible to choose a different name than ``esmvalcore`` for the environment.

.. note::

	  Creating a new Conda environment is often much faster and more reliable than trying to update an existing Conda environment.

Pip installation
-----------------

It is also possible to install ESMValCore from `PyPI <https://pypi.org/project/ESMValCore/>`_.
However, this requires first installing dependencies that are not available on PyPI in some other way.
By far the easiest way to install these dependencies is to use conda_.
For a minimal conda installation (recommended) go to https://conda.io/miniconda.html.

After installing Conda, download
`the file with the list of dependencies <https://raw.githubusercontent.com/ESMValGroup/ESMValCore/main/environment.yml>`_:

.. code-block:: bash

    wget https://raw.githubusercontent.com/ESMValGroup/ESMValCore/main/environment.yml

and install these dependencies into a new conda environment with the command

.. code-block:: bash

    conda env create --name esmvalcore --file environment.yml

Finally, activate the newly created environment

.. code-block:: bash

    conda activate esmvalcore

and install ESMValCore as well as any remaining dependencies with the command:

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

   docker run -e HOME -v "$HOME":"$HOME" -v /data:/data esmvalgroup/esmvalcore:stable -c ~/config-user.yml ~/recipes/recipe_example.yml

with the environmental variable ``$HOME`` available inside the container and the data
in the directories ``$HOME`` and ``/data``, so these can be used to find the configuration file, recipe, and data.

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

   singularity run docker://esmvalgroup/esmvalcore:stable -c ~/config-user.yml ~/recipes/recipe_example.yml

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

   singularity run esmvalcore.sif -c ~/config-user.yml ~/recipes/recipe_example.yml

.. _installation-from-source:

Installation from source
------------------------

.. note::
    If you would like to install the development version of ESMValCore alongside
    ESMValTool, please have a look at
    :ref:`these instructions <esmvaltool:esmvalcore-development-installation>`.

To install from source for development, follow these instructions.

-  `Download and install
   conda <https://conda.io/projects/conda/en/latest/user-guide/install/linux.html>`__
   (this should be done even if the system in use already has a
   preinstalled version of conda, as problems have been reported with
   using older versions of conda)
-  To make the ``conda`` command available, add
   ``source <prefix>/etc/profile.d/conda.sh`` to your ``.bashrc`` file
   and restart your shell. If using (t)csh shell, add
   ``source <prefix>/etc/profile.d/conda.csh`` to your
   ``.cshrc``/``.tcshrc`` file instead.
-  Update conda: ``conda update -y conda``
-  Clone the ESMValCore Git repository:
   ``git clone https://github.com/ESMValGroup/ESMValCore.git``
-  Go to the source code directory: ``cd ESMValCore``
-  Create the esmvalcore conda environment
   ``conda env create --name esmvalcore --file environment.yml``
-  Activate the esmvalcore environment: ``conda activate esmvalcore``
-  Install in development mode: ``pip install -e '.[develop]'``. If you
   are installing behind a proxy that does not trust the usual pip-urls
   you can declare them with the option ``--trusted-host``,
   e.g. ``pip install --trusted-host=pypi.python.org --trusted-host=pypi.org --trusted-host=files.pythonhosted.org -e .[develop]``
-  Test that your installation was successful by running
   ``esmvaltool -h``.

Pre-installed versions on HPC clusters
--------------------------------------

You will find the tool available on HPC clusters and there will be no need to install it
yourself if you are just running diagnostics:

 - CEDA-JASMIN: `esmvaltool` is available on the scientific compute nodes (`sciX.jasmin.ac.uk` where
   `X = 1, 2, 3, 4, 5`) after login and module loading via `module load esmvaltool`; see the helper page at
   `CEDA <https://help.jasmin.ac.uk/article/4955-community-software-esmvaltool>`__ ;
 - DKRZ-Mistral: `esmvaltool` is available on login nodes (`mistral.dkrz.de`) and pre- and post-processing
   nodes (`mistralpp.dkrz.de`) after login and module loading via `module load esmvaltool`; the command
   `module help esmvaltool` provides some information about the module.

.. note::
    If you would like to use pre-installed versions on HPC clusters (currently CEDA-JASMIN and DKRZ-MISTRAL),
    please have a look at
    :ref:`these instructions <esmvaltool:install_on_hpc>`.


.. _condalock-installation-creation:

Installation from the conda lock file
-------------------------------------

A fast conda environment creation is possible using the provided conda lock files. This is a secure alternative
to the installation from source, whenever the conda environment can not be created for some reason. A conda lock file
is an explicit environment file that contains pointers to dependency packages as they are hosted on the Anaconda cloud;
these have frozen version numbers, build hashes, and channel names, parameters established at the time
of the conda lock file creation, so may be obsolete after a while,
but they allow for a robust environment creation while they're still up-to-date.
We regenerate these lock files every 10 days through automatic Pull Requests
(or more frequently, since the automatic generator runs on merges on the main branch too),
so to minimize the risk of dependencies becoming obsolete. Conda environment creation from
a lock file is done just like with any other environment file:

.. code-block:: bash

   conda create --name esmvaltool --file conda-linux-64.lock

The latest, most up-to-date file can always be downloaded directly from the source code
repository, a direct download link can be found `here <https://raw.githubusercontent.com/ESMValGroup/ESMValCore/main/conda-linux-64.lock>`__.

.. note::
   `pip` and `conda` are NOT installed, so you will have to install them in the new environment: use conda-forge as channel): ``conda install -c conda-forge pip`` at the very minimum so we can install `esmvalcore` afterwards.


Creating a conda lock file
--------------------------

We provide a conda lock file for Linux-based operating systems, but if you prefer to
build a conda lock file yourself, install the `conda-lock` package first:

.. code-block:: bash

   conda install -c conda-forge conda-lock

then run

.. code-block:: bash

   conda-lock lock --platform linux-64 -f environment.yml --mamba

(mamba activated for speed) to create a conda lock file for Linux platforms,
or run

.. code-block:: bash

   conda-lock lock --platform osx-64 -f environment.yml --mamba

to create a lock file for OSX platforms. Note however, that using conda lock files on OSX is still problematic!
