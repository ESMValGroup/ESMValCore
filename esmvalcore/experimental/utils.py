"""ESMValCore utilities."""

import re
from pathlib import Path

from esmvalcore._config import DIAGNOSTICS_PATH

from .recipe_info import RecipeInfo


def get_recipes_list() -> list:
    """Return a list of all available recipes."""
    rootdir = Path(DIAGNOSTICS_PATH, 'recipes')
    return list(rootdir.glob('**/*.yml'))


def find_recipes(query: str) -> list:
    """Search for recipes matching the search query or pattern.

    This function will search the recipe description, author list, and
    project information.

    Parameters
    ----------
    query : str
        String to search for, e.g. `find_recipes('righi')` will return all
        matching that author. Can be a `regex` pattern.

    Returns
    -------
    recipes : list
        Returns a list of recipes matching the search query.
    """
    recipes = get_recipes_list()

    query = re.compile(query, flags=re.IGNORECASE)

    result = []

    for recipe in recipes:
        recipe_info = RecipeInfo(recipe)
        match = re.search(query, recipe_info.to_markdown())
        if match:
            result.append(recipe_info)

    return result
