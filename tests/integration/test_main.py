"""
Tests for ESMValTool CLI

Includes a context manager to temporarly modify sys.argv
"""

import sys
import copy
import functools
import pytest
from unittest.mock import patch

from esmvalcore._main import run, ESMValTool, Recipes, Config


def wrapper(f):
    @functools.wraps(f)
    def empty(*args, **kwargs):
        if kwargs:
            raise ValueError(f'Parameters not supported: {kwargs}')
    return empty


class SetArgs(object):
    def __init__(self, *args):
        self._original_args = sys.argv
        self._new_args = args

    def __enter__(self):
        sys.argv = self._new_args

    def __exit__(self, type, value, traceback):
        sys.argv = self._original_args


def test_setargs():
    original = copy.deepcopy(sys.argv)
    with SetArgs('testing', 'working', 'with', 'sys.argv'):
        assert sys.argv == ('testing', 'working', 'with', 'sys.argv')
    assert sys.argv == original


@patch('esmvalcore._main.ESMValTool.version', new=wrapper(ESMValTool.version))
def test_version():
    """Test version command"""
    with SetArgs('esmvaltool', 'version'):
        run()
    with SetArgs('esmvaltool', 'version', '--extra_parameter=asterisk'):
        with pytest.raises(SystemExit):
            run()


@patch('esmvalcore._main.ESMValTool.run', new=wrapper(ESMValTool.run))
def test_run():
    """Test version command"""
    with SetArgs('esmvaltool', 'run', 'recipe.yml'):
        run()


@patch('esmvalcore._main.ESMValTool.run', new=wrapper(ESMValTool.run))
def test_tun_with_config():
    with SetArgs(
            'esmvaltool', 'run', 'recipe.yml', '--config_file', 'config.yml'):
        run()


@patch('esmvalcore._main.ESMValTool.run', new=wrapper(ESMValTool.run))
def test_run_with_max_years():
    with SetArgs(
            'esmvaltool', 'run', 'recipe.yml', '--config_file=config.yml',
            '--max_years=2'):
        run()


@patch('esmvalcore._main.ESMValTool.run', new=wrapper(ESMValTool.run))
def test_run_with_max_datasets():
    with SetArgs('esmvaltool', 'run', 'recipe.yml', '--max_datasets=2'):
        run()


@patch('esmvalcore._main.ESMValTool.run', new=wrapper(ESMValTool.run))
def test_run_with_synda_download():
    with SetArgs('esmvaltool', 'run', 'recipe.yml', '--synda_download=True'):
        run()


@patch('esmvalcore._main.ESMValTool.run', new=wrapper(ESMValTool.run))
def test_run_with_check_level():
    with SetArgs('esmvaltool', 'run', 'recipe.yml', '--check_level=default'):
        run()


@patch('esmvalcore._main.ESMValTool.run', new=wrapper(ESMValTool.run))
def test_run_with_skip_nonexistent():
    with SetArgs('esmvaltool', 'run', 'recipe.yml', '--skip_nonexistent=True'):
        run()


@patch('esmvalcore._main.ESMValTool.run', new=wrapper(ESMValTool.run))
def test_run_with_diagnostics():
    with SetArgs('esmvaltool', 'run', 'recipe.yml', '--diagnostics=[badt]'):
        run()


@patch('esmvalcore._main.ESMValTool.run', new=wrapper(ESMValTool.run))
def test_run_fails_with_other_params():
    with SetArgs('esmvaltool', 'run', 'recipe.yml', '--extra_param=dfa'):
        with pytest.raises(SystemExit):
            run()


@patch('esmvalcore._main.ESMValTool.run', new=wrapper(ESMValTool.run))
def test_run_fails_without_recipe():
    with SetArgs('esmvaltool', 'run'):
        with pytest.raises(SystemExit):
            run()


@patch('esmvalcore._main.Recipes.get', new=wrapper(Recipes.get))
def test_recipes_get():
    """Test version command"""
    with SetArgs('esmvaltool', 'recipes', 'get', 'recipe.yml'):
        run()


@patch('esmvalcore._main.Recipes.get', new=wrapper(Recipes.get))
def test_recipes_get_missing_recipe_fails():
    """Test version command"""
    with SetArgs('esmvaltool', 'recipes', 'get'):
        with pytest.raises(SystemExit):
            run()


@patch('esmvalcore._main.Recipes.list', new=wrapper(Recipes.list))
def test_recipes_list():
    """Test version command"""
    with SetArgs('esmvaltool', 'recipes', 'list'):
        run()


@patch('esmvalcore._main.Recipes.list', new=wrapper(Recipes.list))
def test_recipes_list_do_not_admit_parameters():
    """Test version command"""
    with SetArgs('esmvaltool', 'recipes', 'list', 'parameter'):
        with pytest.raises(SystemExit):
            run()


@patch(
    'esmvalcore._main.Config.get_config_developer',
    new=wrapper(Config.get_config_developer))
def test_get_config_developer():
    """Test version command"""
    with SetArgs('esmvaltool', 'config', 'get_config_developer'):
        run()


@patch(
    'esmvalcore._main.Config.get_config_user',
    new=wrapper(Config.get_config_user))
def test_get_config_user():
    """Test version command"""
    with SetArgs('esmvaltool', 'config', 'get_config_user'):
        run()


@patch(
    'esmvalcore._main.Config.get_config_user',
    new=wrapper(Config.get_config_user))
def test_get_config_user_overwrite():
    """Test version command"""
    with SetArgs(
            'esmvaltool', 'config', 'get_config_user', '--overwrite'):
        run()


def test_get_config_user_target_path():
    """Test version command"""
    with SetArgs(
            'esmvaltool', 'config', 'get_config_user', '--target-path=path'):
        run()


@patch(
    'esmvalcore._main.Config.get_config_user',
    new=wrapper(Config.get_config_user))
def test_get_config_user_path():
    """Test version command"""
    with SetArgs(
            'esmvaltool', 'config', 'get-config-user', '--target_path=path'):
        run()


@patch(
    'esmvalcore._main.Config.get_config_user',
    new=wrapper(Config.get_config_user))
def test_get_config_user_bad_option_fails():
    """Test version command"""
    with SetArgs(
            'esmvaltool', 'config', 'get_config_user', '--bad_option=path'):
        with pytest.raises(SystemExit):
            run()
