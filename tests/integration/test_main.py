"""Tests for ESMValTool CLI.

Includes a context manager to temporarily modify sys.argv
"""
import contextlib
import copy
import functools
import os
import shutil
import sys
from textwrap import dedent
from unittest.mock import patch

import pytest
import yaml
from fire.core import FireExit

from esmvalcore._main import Config, ESMValTool, Recipes, run
from esmvalcore.exceptions import RecipeError


def wrapper(f):
    @functools.wraps(f)
    def empty(*args, **kwargs):
        if kwargs:
            raise ValueError(f'Parameters not supported: {kwargs}')
        return True

    return empty


@contextlib.contextmanager
def arguments(*args):
    backup = sys.argv
    sys.argv = list(args)
    yield
    sys.argv = backup


def test_setargs():
    original = copy.deepcopy(sys.argv)
    with arguments('testing', 'working', 'with', 'sys.argv'):
        assert sys.argv == ['testing', 'working', 'with', 'sys.argv']
    assert sys.argv == original


@patch('esmvalcore._main.ESMValTool.version', new=wrapper(ESMValTool.version))
def test_version():
    """Test version command."""
    with arguments('esmvaltool', 'version'):
        run()
    with arguments('esmvaltool', 'version', '--extra_parameter=asterisk'):
        with pytest.raises(FireExit):
            run()


@patch('esmvalcore._main.ESMValTool.run', new=wrapper(ESMValTool.run))
def test_run():
    """Test version command."""
    with arguments('esmvaltool', 'run', 'recipe.yml'):
        run()


def test_empty_run(tmp_path):
    """Test real run with no diags."""
    recipe_file = tmp_path / "recipe.yml"
    content = dedent("""
        documentation:
          title: Test recipe
          description: This is a test recipe.
          authors:
            - andela_bouwe
          references:
            - contact_authors
            - acknow_project
          projects:
            - c3s-magic
        diagnostics: null
    """)
    recipe_file.write_text(content)
    Config.get_config_user(path=tmp_path)
    log_dir = f'{tmp_path}/esmvaltool_output'
    config_file = f"{tmp_path}/config-user.yml"
    with open(config_file, 'r+') as file:
        config = yaml.safe_load(file)
        config['output_dir'] = log_dir
        yaml.safe_dump(config, file, sort_keys=False)
    with pytest.raises(RecipeError) as exc:
        ESMValTool().run(
            recipe_file, config_file=f"{tmp_path}/config-user.yml")
    assert str(exc.value) == 'The given recipe does not have any diagnostic.'
    log_file = os.path.join(log_dir,
                            os.listdir(log_dir)[0], 'run', 'main_log.txt')
    filled_recipe = os.path.exists(
        log_dir + '/' + os.listdir(log_dir)[0] + '/run/recipe_filled.yml')
    shutil.rmtree(log_dir)

    assert log_file
    assert not filled_recipe


@patch('esmvalcore._main.ESMValTool.run', new=wrapper(ESMValTool.run))
def test_run_with_config():
    with arguments('esmvaltool', 'run', 'recipe.yml', '--config_file',
                   'config.yml'):
        run()


@patch('esmvalcore._main.ESMValTool.run', new=wrapper(ESMValTool.run))
def test_run_with_max_years():
    with arguments('esmvaltool', 'run', 'recipe.yml',
                   '--config_file=config.yml', '--max_years=2'):
        run()


@patch('esmvalcore._main.ESMValTool.run', new=wrapper(ESMValTool.run))
def test_run_with_max_datasets():
    with arguments('esmvaltool', 'run', 'recipe.yml', '--max_datasets=2'):
        run()


@patch('esmvalcore._main.ESMValTool.run', new=wrapper(ESMValTool.run))
def test_run_with_offline():
    with arguments('esmvaltool', 'run', 'recipe.yml', '--offline'):
        run()


@patch('esmvalcore._main.ESMValTool.run', new=wrapper(ESMValTool.run))
def test_run_with_check_level():
    with arguments('esmvaltool', 'run', 'recipe.yml', '--check_level=default'):
        run()


@patch('esmvalcore._main.ESMValTool.run', new=wrapper(ESMValTool.run))
def test_run_with_skip_nonexistent():
    with arguments('esmvaltool', 'run', 'recipe.yml',
                   '--skip_nonexistent=True'):
        run()


@patch('esmvalcore._main.ESMValTool.run', new=wrapper(ESMValTool.run))
def test_run_with_diagnostics():
    with arguments('esmvaltool', 'run', 'recipe.yml', '--diagnostics=[badt]'):
        run()


@patch('esmvalcore._main.ESMValTool.run', new=wrapper(ESMValTool.run))
def test_run_fails_with_other_params():
    with arguments('esmvaltool', 'run', 'recipe.yml', '--extra_param=dfa'):
        with pytest.raises(SystemExit):
            run()


def test_recipes_get(tmp_path, monkeypatch):
    """Test version command."""
    src_recipe = tmp_path / 'recipe.yml'
    src_recipe.touch()
    tgt_dir = tmp_path / 'test'
    tgt_dir.mkdir()
    monkeypatch.chdir(tgt_dir)
    with arguments('esmvaltool', 'recipes', 'get', str(src_recipe)):
        run()
    assert (tgt_dir / 'recipe.yml').is_file()


@patch('esmvalcore._main.Recipes.list', new=wrapper(Recipes.list))
def test_recipes_list():
    """Test version command."""
    with arguments('esmvaltool', 'recipes', 'list'):
        run()


@patch('esmvalcore._main.Recipes.list', new=wrapper(Recipes.list))
def test_recipes_list_do_not_admit_parameters():
    """Test version command."""
    with arguments('esmvaltool', 'recipes', 'list', 'parameter'):
        with pytest.raises(FireExit):
            run()


@patch('esmvalcore._main.Config.get_config_developer',
       new=wrapper(Config.get_config_developer))
def test_get_config_developer():
    """Test version command."""
    with arguments('esmvaltool', 'config', 'get_config_developer'):
        run()


@patch('esmvalcore._main.Config.get_config_user',
       new=wrapper(Config.get_config_user))
def test_get_config_user():
    """Test version command."""
    with arguments('esmvaltool', 'config', 'get_config_user'):
        run()


def test_get_config_user_path(tmp_path):
    """Test version command."""
    with arguments('esmvaltool', 'config', 'get_config_user',
                   f'--path={tmp_path}'):
        run()
    assert (tmp_path / 'config-user.yml').is_file()


def test_get_config_user_overwrite(tmp_path):
    """Test version command."""
    config_user = tmp_path / 'config-user.yml'
    config_user.touch()
    with arguments('esmvaltool', 'config', 'get_config_user',
                   f'--path={tmp_path}', '--overwrite'):
        run()


@patch('esmvalcore._main.Config.get_config_user',
       new=wrapper(Config.get_config_user))
def test_get_config_user_bad_option_fails():
    """Test version command."""
    with arguments('esmvaltool', 'config', 'get_config_user',
                   '--bad_option=path'):
        with pytest.raises(FireExit):
            run()
