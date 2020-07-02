"""
Tests for ESMValTool CLI

Includes a context manager to temporarly modify sys.argv
"""

import contextlib
import copy
import functools
import sys
from unittest.mock import patch

import pytest
from fire.core import FireExit

from esmvalcore._main import Config, ESMValTool, Recipes, run


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
    """Test version command"""
    with arguments('esmvaltool', 'version'):
        run()
    with arguments('esmvaltool', 'version', '--extra_parameter=asterisk'):
        with pytest.raises(FireExit):
            run()


@patch('esmvalcore._main.ESMValTool.run', new=wrapper(ESMValTool.run))
def test_run():
    """Test version command"""
    with arguments('esmvaltool', 'run', 'recipe.yml'):
        run()


@patch('esmvalcore._main.ESMValTool.run', new=wrapper(ESMValTool.run))
def test_tun_with_config():
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
def test_run_with_synda_download():
    with arguments('esmvaltool', 'run', 'recipe.yml', '--synda_download=True'):
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
    """Test version command"""
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
    """Test version command"""
    with arguments('esmvaltool', 'recipes', 'list'):
        run()


@patch('esmvalcore._main.Recipes.list', new=wrapper(Recipes.list))
def test_recipes_list_do_not_admit_parameters():
    """Test version command"""
    with arguments('esmvaltool', 'recipes', 'list', 'parameter'):
        with pytest.raises(FireExit):
            run()


@patch('esmvalcore._main.Config.get_config_developer',
       new=wrapper(Config.get_config_developer))
def test_get_config_developer():
    """Test version command"""
    with arguments('esmvaltool', 'config', 'get_config_developer'):
        run()


@patch('esmvalcore._main.Config.get_config_user',
       new=wrapper(Config.get_config_user))
def test_get_config_user():
    """Test version command"""
    with arguments('esmvaltool', 'config', 'get_config_user'):
        run()


def test_get_config_user_path(tmp_path):
    """Test version command"""
    with arguments('esmvaltool', 'config', 'get_config_user',
                   f'--path={tmp_path}'):
        run()
    assert (tmp_path / 'config-user.yml').is_file()


def test_get_config_user_overwrite(tmp_path):
    """Test version command"""
    config_user = tmp_path / 'config-user.yml'
    config_user.touch()
    with arguments('esmvaltool', 'config', 'get_config_user',
                   f'--path={tmp_path}', '--overwrite'):
        run()


@patch('esmvalcore._main.Config.get_config_user',
       new=wrapper(Config.get_config_user))
def test_get_config_user_bad_option_fails():
    """Test version command"""
    with arguments('esmvaltool', 'config', 'get_config_user',
                   '--bad_option=path'):
        with pytest.raises(FireExit):
            run()
