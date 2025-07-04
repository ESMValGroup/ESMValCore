"""ESMValCore experimental API module."""

import logging
import sys

from esmvalcore.config import CFG

from ._warnings import warnings  # prints experimental API warning
from .recipe import Recipe
from .utils import RecipeList, get_all_recipes, get_recipe

logging.basicConfig(
    format="%(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)

__all__ = [
    "CFG",
    "get_all_recipes",
    "get_recipe",
    "Recipe",
    "RecipeList",
    "warnings",
]
