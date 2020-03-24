.. _running:

Running
*******

The ESMValCore package provides the ``esmvaltool`` command line tool, which can be used to run a :doc:`recipe <../recipe/index>`.

The minimal arguments that must be provided to the command are the path to the :ref:`user configuration file<config>` and the path to a recipe.

.. code:: bash

	esmvaltool -c /path/to/config-user.yml recipe_python.yml

To get help on additional commands, please use

.. code:: bash

	esmvaltool -h
