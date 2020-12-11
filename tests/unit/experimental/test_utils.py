import pytest

from esmvalcore.experimental.recipe_info import RecipeInfo
from esmvalcore.experimental.utils import get_all_recipes, get_recipe
"""
The behaviour of these tests are somewhat unpredictable, depending on
whether ESMValTool is installed or not, which defines the location of
``DIAGNOSTICS_PATH``.
"""


@pytest.mark.xfail('FileNotFoundError')
def test_get_recipe():
    """Get single recipe."""
    recipe = get_recipe('examples/recipe_python.yml')
    assert isinstance(recipe, RecipeInfo)


def test_get_all_recipes():
    """Get all recipes."""
    recipes = get_all_recipes()
    assert isinstance(recipes, list)
