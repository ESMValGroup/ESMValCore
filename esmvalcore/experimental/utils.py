"""ESMValCore utilities."""

import re
from pathlib import Path

from esmvalcore._config import DIAGNOSTICS

from .recipe import Recipe


class RecipeList(list):
    """Container for recipes."""
    def find(self, query: str):
        """Search for recipes matching the search query or pattern.

        Searches in the description, authors and project information fields.
        All matches are returned.

        Parameters
        ----------
        query : str
            String to search for, e.g. ``find_recipes('righi')`` will return
            all matching that author. Can be a `regex` pattern.

        Returns
        -------
        RecipeList
            List of recipes matching the search query.
        """
        query = re.compile(query, flags=re.IGNORECASE)

        matches = RecipeList()

        for recipe in self:
            match = re.search(query, str(recipe))
            if match:
                matches.append(recipe)

        return matches


def get_all_recipes(subdir: str = None) -> list:
    """Return a list of all available recipes.

    Parameters
    ----------
    subdir : str
        Sub-directory of the ``DIAGNOSTICS.path`` to look for
        recipes, e.g. ``get_all_recipes(subdir='examples')``.

    Returns
    -------
    RecipeList
        List of available recipes
    """
    if not subdir:
        subdir = '**'
    rootdir = DIAGNOSTICS.recipes
    files = rootdir.glob(f'{subdir}/*.yml')
    return RecipeList(Recipe(file) for file in files)


def get_recipe(name: str) -> 'Recipe':
    """Get a recipe by its name.

    The function looks first in the local directory, and second in the
    repository defined by the diagnostic path. The recipe name can be
    specified with or without extension. The first match will be returned.

    Parameters
    ----------
    name : str, pathlike
        Name of the recipe file, i.e. ``examples/recipe_python.yml``

    Returns
    -------
    Recipe
        Instance of :obj:`Recipe` which can be used to inspect and run
        the recipe.

    Raises
    ------
    FileNotFoundError
        If the name cannot be resolved to a recipe file.
    """
    locations = Path(), DIAGNOSTICS.recipes

    if isinstance(name, Path):
        filenames = (name, )
    else:
        filenames = (name, name + '.yml')

    for location in locations:
        for filename in filenames:
            try_path = Path(location, filename).expanduser()
            if try_path.exists():
                return Recipe(try_path)

    raise FileNotFoundError(f'Could not find `{name}` in {locations}.')
