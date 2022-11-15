.. _api_utils:



Utils
=====

This section describes the :py:class:`~esmvalcore.experimental.utils` submodule of the API (:py:mod:`esmvalcore.experimental`).


Finding recipes
***************

One of the first thing we may want to do, is to simply get one of the recipes available in ``ESMValTool``

If you already know which recipe you want to load, call :py:func:`~esmvalcore.experimental.utils.get_recipe`.

.. code-block:: python

    from esmvalcore.experimental import get_recipe
    >>> get_recipe('examples/recipe_python')
    Recipe('Recipe python')

Call the :py:func:`~esmvalcore.experimental.utils.get_all_recipes` function to get a list of all available recipes.

.. code-block:: python

    >>> from esmvalcore.experimental import get_all_recipes
    >>> recipes = get_all_recipes()
    >>> recipes
    [Recipe('Recipe perfmetrics cmip5 4cds'),
     Recipe('Recipe martin18grl'),
     ...
     Recipe('Recipe wflow'),
     Recipe('Recipe pcrglobwb')]

To search for a specific recipe, you can use the :py:meth:`~esmvalcore.experimental.utils.RecipeList.find` method. This takes a search query that looks through the recipe metadata and returns any matches. The query can be a regex pattern, so you can make it as complex as you like.

.. code-block:: python

    >>> results = recipes.find('climwip')
    [Recipe('Recipe climwip')]

The recipes are loaded in a :py:class:`~esmvalcore.experimental.recipe.Recipe` object, which knows about the documentation, authors, project, and related references of the recipe. It resolves all the tags, so that it knows which institute an author belongs to and which references are associated with the recipe.

This means you can search for something like this:

.. code-block:: python

    >>> recipes.find('Geophysical Research Letters')
    [Recipe('Recipe martin18grl'),
     Recipe('Recipe climwip'),
     Recipe('Recipe ecs constraints'),
     Recipe('Recipe ecs scatter'),
     Recipe('Recipe ecs'),
     Recipe('Recipe seaice')]


API reference
*************

.. automodule:: esmvalcore.experimental.utils
    :no-inherited-members:
    :no-show-inheritance:
