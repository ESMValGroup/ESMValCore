"""ESMValCore utilities."""

import re
from pathlib import Path

from esmvalcore._config import DIAGNOSTICS_PATH

from .recipe_info import RecipeInfo


class RecipeList(list):
    """Container for recipes."""
    def find(self, query: str):
        """Search for recipes matching the search query or pattern.

        This function will search the recipe description, author list, and
        project information.

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
        Sub-directory of the ``DIAGNOSTICS_PATH`` to look for
        recipes, e.g. ``get_all_recipes(subdir='examples')``.

    Returns
    -------
    RecipeList
        List of available recipes
    """
    if not subdir:
        subdir = '**'
    rootdir = Path(DIAGNOSTICS_PATH, 'recipes')
    files = rootdir.glob(f'{subdir}/*.yml')
    return RecipeList(RecipeInfo(file) for file in files)


def get_recipe(name: str) -> 'RecipeInfo':
    """Get a recipe by its name.

    Parameters
    ----------
    name : str
        Name of the recipe, i.e. ``examples/recipe_python.yml``

    Returns
    -------
    RecipeInfo
    """
    locations = Path(), Path(DIAGNOSTICS_PATH, 'recipes')
    filenames = name, name + '.yml'

    for location in locations:
        for filename in filenames:
            try_path = Path(location, filename)
            if try_path.exists():
                return RecipeInfo(try_path)

    raise FileNotFoundError(f'Could not find `{name}')
