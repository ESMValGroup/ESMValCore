.. _api_recipe_info:

Recipes
=======

This section describes the :py:mod:`esmvalcore.experimental.recipe_info` submodule of the API.

Recipe metadata
***************

:py:class:`esmvalcore.experimental.recipe_info.RecipeInfo` info is a class that holds metadata from a recipe.

.. code-block:: python

    >>> RecipeInfo('path/to/recipe_python.yml')
    recipe = RecipeInfo('Recipe Python')

Printing the recipe will give a nice overview of the recipe:

.. code-block:: python

    >>> print(recipe)
    ## Recipe python

    Example recipe that plots a map and timeseries of temperature.

    ### Authors
     - Bouwe Andela (NLeSC, Netherlands; https://orcid.org/0000-0001-9005-8940)
     - Mattia Righi (DLR, Germany; https://orcid.org/0000-0003-3827-5950)

    ### Maintainers
     - Manuel Schlund (DLR, Germany; https://orcid.org/0000-0001-5251-0158)

    ### Projects
     - DLR project ESMVal
     - Copernicus Climate Change Service 34a Lot 2 (MAGIC) project

    ### References
     - Please acknowledge the project(s).

Running a recipe
****************

To run the recipe, call the :py:meth:`esmvalcore.experimental.RecipeInfo.run` method.

.. code-block:: python

    >>> output = recipe_info.run()
    <log messages>

By default, a new :py:class:`esmvalcore.experimental.config.Session` is automatically created, so that data are never overwritten.
Data are stored in the ``esmvaltool_output`` directory specified in the config.
Sessions can also be explicitly specified.

.. code-block:: python

    >>> from esmvalcore.experimental import CFG
    >>> session = CFG.start_session('my_session')
    >>> output = recipe_info.run(session)
    <log messages>

:py:meth:`esmvalcore.experimental.RecipeInfo.run` returns an object that contains the locations of the data and figures (not implemented yet).
