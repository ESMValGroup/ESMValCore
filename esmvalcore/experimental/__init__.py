"""ESMValCore experimental API module."""

import logging
import sys

from ._warnings import warnings  # prints experimental API warning
from .config import CFG
from .recipe_info import RecipeInfo
from .utils import RecipeList, get_all_recipes, get_recipe

logging.basicConfig(format='%(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

__all__ = [
    'CFG',
    'get_all_recipes',
    'get_recipe',
    'RecipeInfo',
    'RecipeList',
    'warnings',
]
