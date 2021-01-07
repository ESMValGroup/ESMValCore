"""Tests running a recipe using sample data.

Runs recipes using :meth:`esmvalcore.experimental.RecipeInfo.run`.
"""

from pathlib import Path

import pytest

from esmvalcore.experimental import CFG, RecipeInfo, get_recipe

esmvaltool_sample_data = pytest.importorskip("esmvaltool_sample_data")

CFG.update(esmvaltool_sample_data.get_rootpaths())
CFG['max_parallel_tasks'] = 1


@pytest.fixture
def recipe():
    recipe = get_recipe(Path(__file__).with_name('recipe_api_test.yml'))
    return recipe


@pytest.mark.use_sample_data
def test_run_recipe(recipe, tmp_path):
    """Test running a basic recipe using sample data."""
    CFG['output_dir'] = tmp_path

    assert isinstance(recipe, RecipeInfo)

    output = recipe.run()

    assert not output  # output is not yet defined
