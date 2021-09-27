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

To automatically download the files required to run a recipe from ESGF, set
``offline`` to ``false`` in the :ref:`user configuration file`
or run the tool with the command

.. code:: bash

    esmvaltool run --offline=False recipe_python.yml

This feature is available for projects that are hosted on the ESGF, i.e.
CMIP3, CMIP5, CMIP6, CORDEX, and obs4MIPs.

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
Note that diagnostics may require specific combinations of available data, so
use the above two flags at your own risk and for testing purposes only.

To run a recipe, even if some datasets are not available, use

.. code:: bash

    esmvaltool run --skip_nonexistent=True recipe_python.yml

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
