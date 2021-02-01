.. _api_recipe:

Recipes
=======

This section describes the :py:mod:`~esmvalcore.experimental.recipe` submodule of the API (:py:mod:`esmvalcore.experimental`).

Recipe metadata
***************

:py:class:`~esmvalcore.experimental.recipe.Recipe` is a class that holds metadata from a recipe.

.. code-block:: python

    >>> Recipe('path/to/recipe_python.yml')
    recipe = Recipe('Recipe Python')

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

To run the recipe, call the :py:meth:`~esmvalcore.experimental.recipe.Recipe.run` method.

.. code-block:: python

    >>> output = recipe.run()
    <log messages>

By default, a new :py:class:`~esmvalcore.experimental.config.Session` is automatically created, so that data are never overwritten.
Data are stored in the ``esmvaltool_output`` directory specified in the config.
Sessions can also be explicitly specified.

.. code-block:: python

    >>> from esmvalcore.experimental import CFG
    >>> session = CFG.start_session('my_session')
    >>> output = recipe.run(session)
    <log messages>

:py:meth:`~esmvalcore.experimental.recipe.Recipe.run` returns an dictionary of objects that can be used to inspect
the output of the recipe. The output is an instance of :py:class:`~esmvalcore.experimental.recipe_output.ImageFile` or
:py:class:`~esmvalcore.experimental.recipe_output.DataFile` depending on its type.

For working with recipe output, see: :ref:`api_recipe_output`.

Running a single diagnostic or preprocessor task
************************************************

The python example recipe contains 5 tasks:

Preprocessors:

- ``timeseries/tas_amsterdam``
- ``timeseries/script1``
- ``map/tas``

Diagnostics:

- ``timeseries/tas_global``
- ``map/script1``

To run a single diagnostic or preprocessor, the name of the task can be passed
as an argument to :py:meth:`~esmvalcore.experimental.recipe.Recipe.run`. If a diagnostic
is passed, all ancestors will automatically be run too.

.. code-block:: python

    >>> output = recipe.run('map/script1')
    >>> output
    map/script1:
      DataFile('CMIP5_CanESM2_Amon_historical_r1i1p1_tas_2000-2000.nc')
      DataFile('CMIP6_BCC-ESM1_Amon_historical_r1i1p1f1_tas_2000-2000.nc')
      ImageFile('CMIP5_CanESM2_Amon_historical_r1i1p1_tas_2000-2000.png')
      ImageFile('CMIP6_BCC-ESM1_Amon_historical_r1i1p1f1_tas_2000-2000.png')

It is also possible to run a single preprocessor task:

.. code-block:: python

    >>> output = recipe.run('map/tas')
    >>> output
    map/tas:
      DataFile('CMIP5_CanESM2_Amon_historical_r1i1p1_tas_2000-2000.nc')
      DataFile('CMIP6_BCC-ESM1_Amon_historical_r1i1p1f1_tas_2000-2000.nc')


API reference
*************

.. automodule:: esmvalcore.experimental.recipe
