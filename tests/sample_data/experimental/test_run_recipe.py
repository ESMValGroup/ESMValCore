"""Tests running a recipe using sample data.

Runs recipes using :meth:`esmvalcore.experimental.Recipe.run`.
"""

from pathlib import Path

import iris
import pytest

from esmvalcore.experimental import CFG, Recipe, get_recipe
from esmvalcore.experimental.recipe_output import DataItem

esmvaltool_sample_data = pytest.importorskip("esmvaltool_sample_data")

CFG.update(esmvaltool_sample_data.get_rootpaths())
CFG['max_parallel_tasks'] = 1


@pytest.fixture
def recipe():
    recipe = get_recipe(Path(__file__).with_name('recipe_api_test.yml'))
    return recipe


@pytest.mark.use_sample_data
def test_run_recipe(recipe, tmp_path):
    """Test running a basic recipe using sample data.

    Recipe contains no provenance and no diagnostics.
    """
    CFG['output_dir'] = tmp_path

    assert isinstance(recipe, Recipe)

    output = recipe.run()

    assert len(output) > 0
    assert isinstance(output, dict)

    for task, items in output.items():
        assert len(items) > 0
        for item in items:
            assert isinstance(item, DataItem)
            assert item.filename.exists()

            cube = item.load_iris()
            assert isinstance(cube, iris.cube.CubeList)
