import pytest

from esmvalcore._config import DIAGNOSTICS, TAGS
from esmvalcore.experimental import get_recipe

pytest.importorskip(
    'esmvaltool',
    reason='The behaviour of these tests depends on what ``DIAGNOSTICS.path``'
    'points to. This is defined by a forward-reference to ESMValTool, which'
    'is not installed in the CI, but likely to be available in a developer'
    'or user installation.')


def test_recipe():
    """Coverage test for Recipe."""
    TAGS.set_tag_values(DIAGNOSTICS.load_tags())

    recipe = get_recipe('examples/recipe_python')

    assert isinstance(repr(recipe), str)
    assert isinstance(str(recipe), str)
    assert isinstance(recipe.render(), str)
