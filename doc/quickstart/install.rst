Installation
============

Conda installation
------------------

In order to install the Conda package, you will need to install conda first.
For a minimal conda installation go to https://conda.io/miniconda.html.
It is recommended that you always use the latest version of conda, as problems have been reported when trying to use older versions.

Once you have installed conda, you can install ESMValCore by running:

.. code-block:: bash


    conda install -c esmvalgroup -c conda-forge esmvalcore


Development installation
------------------------

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
   ``git clone git@github.com:ESMValGroup/ESMValCore``
-  Go to the source code directory: ``cd ESMValCore``
-  Create the esmvalcore conda environment
   ``conda env create --name esmvalcore --file environment.yml``
-  Activate the esmvalcore environment: ``conda activate esmvalcore``
-  Install in development mode: ``pip install -e '.[develop]'``. If you
   are installing behind a proxy that does not trust the usual pip-urls
   you can declare them with the option ``--trusted-host``,
   e.g. \ ``pip install --trusted-host=pypi.python.org --trusted-host=pypi.org --trusted-host=files.pythonhosted.org -e .[develop]``
-  Test that your installation was successful by running
   ``esmvaltool -h``.

