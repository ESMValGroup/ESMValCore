"""Tests running a recipe using sample data.

Runs recipes using :meth:`esmvalcore.experimental.Recipe.run`.
"""

import logging
import os
from contextlib import contextmanager
from pathlib import Path

import iris
import pytest

import esmvalcore._task
from esmvalcore.config._config_object import CFG_DEFAULT
from esmvalcore.config._diagnostics import TAGS
from esmvalcore.exceptions import RecipeError
from esmvalcore.experimental import CFG, Recipe, get_recipe
from esmvalcore.experimental.recipe_output import (
    DataFile,
    RecipeOutput,
    TaskOutput,
)

esmvaltool_sample_data = pytest.importorskip("esmvaltool_sample_data")

AUTHOR_TAGS = {
    'authors': {
        'doe_john': {
            'name': 'Doe, John',
            'institute': 'Testing',
            'orcid': 'https://orcid.org/0000-0000-0000-0000',
        }
    }
}


@pytest.fixture(autouse=True)
def get_mock_distributed_client(monkeypatch):
    """Mock `get_distributed_client` to avoid starting a Dask cluster."""

    @contextmanager
    def get_distributed_client():
        yield None

    monkeypatch.setattr(
        esmvalcore._task,
        'get_distributed_client',
        get_distributed_client,
    )


@pytest.fixture
def recipe():
    recipe = get_recipe(Path(__file__).with_name('recipe_api_test.yml'))
    return recipe


@pytest.mark.use_sample_data
@pytest.mark.parametrize('ssh', (True, False))
@pytest.mark.parametrize('task', (None, 'example/ta'))
def test_run_recipe(monkeypatch, task, ssh, recipe, tmp_path, caplog):
    """Test running a basic recipe using sample data.

    Recipe contains no provenance and no diagnostics.
    """
    caplog.set_level(logging.INFO)
    caplog.clear()
    if ssh:
        monkeypatch.setitem(os.environ, 'SSH_CONNECTION', '0.0 0 1.1 1')
    else:
        monkeypatch.delitem(os.environ, 'SSH_CONNECTION', raising=False)

    TAGS.set_tag_values(AUTHOR_TAGS)

    assert isinstance(recipe, Recipe)
    assert isinstance(recipe._repr_html_(), str)

    sample_data_config = esmvaltool_sample_data.get_rootpaths()
    monkeypatch.setitem(CFG, 'rootpath', sample_data_config['rootpath'])
    monkeypatch.setitem(CFG, 'drs', {'CMIP6': 'SYNDA'})
    session = CFG.start_session(recipe.path.stem)
    session.clear()
    session.update(CFG_DEFAULT)
    session['output_dir'] = tmp_path / 'esmvaltool_output'
    session['max_parallel_tasks'] = 1
    session['remove_preproc_dir'] = False

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

    msg = "It looks like you are connected to a remote machine via SSH."
    if ssh:
        assert msg in caplog.text
    else:
        assert msg not in caplog.text


@pytest.mark.use_sample_data
def test_run_recipe_diagnostic_failing(monkeypatch, recipe, tmp_path):
    """Test running a single diagnostic using sample data.

    Recipe contains no provenance and no diagnostics.
    """
    TAGS.set_tag_values(AUTHOR_TAGS)

    monkeypatch.setitem(CFG, 'output_dir', tmp_path)

    session = CFG.start_session(recipe.path.stem)

    with pytest.raises(RecipeError):
        task = 'example/non-existant'
        _ = recipe.run(task, session)
