.. _running:

Running
*******

The ESMValCore package provides the ``esmvaltool`` command line tool, which can
be used to run a :doc:`recipe <../recipe/index>`.

To run a recipe, call ``esmvaltool run`` with the desired recipe:

.. code:: bash

	esmvaltool run recipe_python.yml

If the configuration file is not in the default location
``~/.esmvaltool/config-user.yml``, you can pass its path explicitly:

.. code:: bash

	esmvaltool run --config_file /path/to/config-user.yml recipe_python.yml

It is also possible to explicitly change values from the config file using flags:

.. code:: bash

	esmvaltool run --argument_name argument_value recipe_python.yml

To control the strictness of the CMOR checker, use the flag ``--check_level``:

.. code:: bash

	esmvaltool run --check_level=relaxed recipe_python.yml

Possible values are:

  - `ignore`: all errors will be reported as warnings
  - `relaxed`: only fail if there are critical errors
  - `default`: fail if there are any errors
  - `strict`: fail if there are any warnings


To run a reduced version of the recipe, usually for testing purpose you can use

.. code:: bash

	esmvaltool run --max_datasets=NDATASETS --max_years=NYEARS recipe_python.yml

In this case, the recipe will limit the number of datasets per variable to
NDATASETS and the total amount of years loaded to NYEARS. They can also be used
separately.

To run a recipe, even if some datasets are not available, use

.. code:: bash

    esmvaltool run --skip_nonexistent=True recipe_python.yml


If Synda is installed (see http://prodiguer.github.io/synda/), it is possible
to use it to automatically download the requested data from ESGF if it is not
available locally:

.. code:: bash

    esmvaltool run --synda_download=True recipe_python.yml

It is also possible to select only specific diagnostics to be run. To tun only
one, just specify its name. To provide more than one diagnostic to filter use
the syntax 'diag1 diag2/script1' or '("diag1", "diag2/script1")' and pay
attention to the quotes.

.. code:: bash

    esmvaltool run --diagnostics=diagnostic1 recipe_python.yml



To get help on additional commands, please use

.. code:: bash

	esmvaltool --help



.. note::

	ESMValTool command line interface is created using the Fire python package.
	This package supports the creation of completion scripts for the Bash and
	Fish shells. Go to https://google.github.io/python-fire/using-cli/#python-fires-flags
	to learn how to set up them.
