"""ESMValCore experimental API module."""

from esmvalcore.config import CFG

from ._logging import _log_to_stdout
from ._warnings import warnings  # prints experimental API warning
from .recipe import Recipe
from .utils import RecipeList, get_all_recipes, get_recipe

_log_to_stdout()

__all__ = [
    "CFG",
    "Recipe",
    "RecipeList",
    "get_all_recipes",
    "get_recipe",
    "warnings",
]
