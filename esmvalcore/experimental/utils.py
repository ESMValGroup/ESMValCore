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
            String to search for, e.g. `find_recipes('righi')` will return all
            matching that author. Can be a `regex` pattern.

        Returns
        -------
        recipes : RecipeList
            Returns a list of recipes matching the search query.
        """
        query = re.compile(query, flags=re.IGNORECASE)

        matches = RecipeList()

        for recipe in self:
            match = re.search(query, str(recipe))
            if match:
                matches.append(recipe)

        return matches


def get_recipes() -> list:
    """Return a list of all available recipes."""
    rootdir = Path(DIAGNOSTICS_PATH, 'recipes')
    files = rootdir.glob('**/*.yml')
    return RecipeList(RecipeInfo(file) for file in files)
