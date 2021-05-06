.. _recipes:

Working with the installed recipes
**********************************

Although ESMValTool can be used just to simplify the management of data
and the creation of your own analysis code, one of its main strengths is the
continuously growing set of diagnostics and metrics that it directly provides to
the user. These metrics and diagnostics are provided as a set of preconfigured
recipes that users can run or customize for their own analysis.
The latest list of available recipes can be found :ref:`here <esmvaltool:recipes>`.

In order to make the management of these installed recipes easier, ESMValTool
provides the ``recipes`` command group with utilities that help the users in
discovering and customizing the provided recipes.

The first command in this group allows users to get the complete list of installed
recipes printed to the console:

.. code:: bash

	esmvaltool recipes list

If the user then wants to explore any one of this recipes, they can be printed
using the following command

.. code:: bash

	esmvaltool recipes show recipe_name.yml

And finally, to get a local copy that can then be customized and run, users can
use the following command

.. code:: bash

	esmvaltool recipes get recipe_name.yml
