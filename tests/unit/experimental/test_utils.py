import pytest

from esmvalcore._config import DIAGNOSTICS, TAGS
from esmvalcore.experimental.recipe import Recipe
from esmvalcore.experimental.utils import (
    RecipeList,
    get_all_recipes,
    get_recipe,
)

pytest.importorskip(
    'esmvaltool',
    reason='The behaviour of these tests depends on what ``DIAGNOSTICS.path``'
    'points to. This is defined by a forward-reference to ESMValTool, which'
    'is not installed in the CI, but likely to be available in a developer'
    'or user installation.')


def test_get_recipe():
    """Get single recipe."""
    recipe = get_recipe('examples/recipe_python.yml')
    assert isinstance(recipe, Recipe)


def test_get_all_recipes():
    """Get all recipes."""
    recipes = get_all_recipes()
    assert isinstance(recipes, list)


def test_recipe_list_find():
    """Get all recipes."""
    TAGS.set_tag_values(DIAGNOSTICS.load_tags())

    recipes = get_all_recipes(subdir='examples')

    assert isinstance(recipes, RecipeList)

    result = recipes.find('python')

    assert isinstance(result, RecipeList)
