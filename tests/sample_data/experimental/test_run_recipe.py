"""Tests running a recipe using sample data.

Runs recipes using :meth:`esmvalcore.experimental.Recipe.run`.
"""

from pathlib import Path

import iris
import pytest

from esmvalcore._recipe import RecipeError
from esmvalcore.experimental import CFG, Recipe, get_recipe
from esmvalcore.experimental.recipe_output import (
    DataFile,
    RecipeOutput,
    TaskOutput,
)

esmvaltool_sample_data = pytest.importorskip("esmvaltool_sample_data")

CFG.update(esmvaltool_sample_data.get_rootpaths())
CFG['max_parallel_tasks'] = 1


@pytest.fixture
def recipe():
    recipe = get_recipe(Path(__file__).with_name('recipe_api_test.yml'))
    return recipe


@pytest.mark.use_sample_data
@pytest.mark.parametrize('task', (None, 'example/ta'))
def test_run_recipe(task, recipe, tmp_path):
    """Test running a basic recipe using sample data.

    Recipe contains no provenance and no diagnostics.
    """
    CFG['output_dir'] = tmp_path

    assert isinstance(recipe, Recipe)

    output = recipe.run(task=task)

    assert len(output) > 0
    assert isinstance(output, RecipeOutput)

    for task, task_output in output.items():
        assert isinstance(task_output, TaskOutput)
        assert len(task_output) > 0

        for data_file in task_output.data_files:
            assert isinstance(data_file, DataFile)
            assert data_file.filename.exists()

            cube = data_file.load_iris()
            assert isinstance(cube, iris.cube.CubeList)


@pytest.mark.use_sample_data
def test_run_recipe_diagnostic_failing(recipe, tmp_path):
    """Test running a single diagnostic using sample data.

    Recipe contains no provenance and no diagnostics.
    """
    CFG['output_dir'] = tmp_path

    with pytest.raises(RecipeError):
        task = 'example/FAIL'
        _ = recipe.run(task)
