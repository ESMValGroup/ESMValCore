"""ESMValCore experimental API module."""

from ._warnings import warnings

warnings.warn(
    '\n  Thank you for trying out the new ESMValCore API.'
    '\n  Note that this API is experimental and may be subject to change.'
    '\n  More info: https://github.com/ESMValGroup/ESMValCore/issues/498', )

from .config import CFG  # noqa: E402
from .recipe_info import RecipeInfo  # noqa: E402
from .utils import find_recipes, get_recipes_list  # noqa: E402

__all__ = [
    'CFG',
    'find_recipes',
    'get_recipes_list',
    'find_recipes',
    'RecipeInfo',
]
