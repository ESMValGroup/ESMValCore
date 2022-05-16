"""Tests running a recipe using sample data.

Runs recipes using :meth:`esmvalcore.experimental.Recipe.run`.
"""

from pathlib import Path

import iris
import pytest

from esmvalcore._config import TAGS
from esmvalcore.exceptions import RecipeError
from esmvalcore.experimental import CFG, Recipe, get_recipe
from esmvalcore.experimental.recipe_output import (
    DataFile,
    RecipeOutput,
    TaskOutput,
)

esmvaltool_sample_data = pytest.importorskip("esmvaltool_sample_data")

CFG.update(esmvaltool_sample_data.get_rootpaths())
CFG['drs']['CMIP6'] = 'SYNDA'
CFG['max_parallel_tasks'] = 1
CFG['remove_preproc_dir'] = False

AUTHOR_TAGS = {
    'authors': {
        'doe_john': {
            'name': 'Doe, John',
            'institute': 'Testing',
            'orcid': 'https://orcid.org/0000-0000-0000-0000',
        }
    }
}


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
    TAGS.set_tag_values(AUTHOR_TAGS)

    CFG['output_dir'] = tmp_path

    assert isinstance(recipe, Recipe)
    assert isinstance(recipe._repr_html_(), str)

    session = CFG.start_session(recipe.path.stem)

    output = recipe.run(task=task, session=session)

    assert len(output) > 0
    assert isinstance(output, RecipeOutput)
    assert (output.session.session_dir / 'index.html').exists()

    assert (output.session.run_dir / output.info.filename).exists()
    assert isinstance(output.read_main_log(), str)
    assert isinstance(output.read_main_log_debug(), str)

    for task, task_output in output.items():
        assert isinstance(task_output, TaskOutput)
        assert len(task_output) > 0

        for data_file in task_output.data_files:
            assert isinstance(data_file, DataFile)
            assert data_file.path.exists()

            cube = data_file.load_iris()
            assert isinstance(cube, iris.cube.CubeList)


@pytest.mark.use_sample_data
def test_run_recipe_diagnostic_failing(recipe, tmp_path):
    """Test running a single diagnostic using sample data.

    Recipe contains no provenance and no diagnostics.
    """
    TAGS.set_tag_values(AUTHOR_TAGS)

    CFG['output_dir'] = tmp_path

    session = CFG.start_session(recipe.path.stem)

    with pytest.raises(RecipeError):
        task = 'example/non-existant'
        _ = recipe.run(task, session)
