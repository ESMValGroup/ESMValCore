import logging
import os
import pathlib
from unittest import mock

import pytest

import esmvalcore._config
import esmvalcore._main
import esmvalcore._task
import esmvalcore.esgf
from esmvalcore import __version__
from esmvalcore._main import HEADER, ESMValTool
from esmvalcore.cmor.check import CheckLevels

LOGGER = logging.getLogger(__name__)


@pytest.mark.parametrize('cmd_offline', [None, True, False])
@pytest.mark.parametrize('cfg_offline', [True, False])
def test_run(mocker, tmp_path, cmd_offline, cfg_offline):

    output_dir = tmp_path / 'output_dir'
    recipe = tmp_path / 'recipe_test.yml'
    recipe.touch()
    offline = cmd_offline is True or (cmd_offline is None
                                      and cfg_offline is True)

    # Minimal config-user.yml for ESMValTool run function.
    cfg = {
        'config_file': tmp_path / '.esmvaltool' / 'config-user.yml',
        'log_level': 'info',
        'offline': cfg_offline,
        'preproc_dir': str(output_dir / 'preproc_dir'),
        'run_dir': str(output_dir / 'run_dir'),
    }

    # Expected configuration after updating from command line.
    reference = dict(cfg)
    reference.update({
        'check_level': CheckLevels.DEFAULT,
        'diagnostics': set(),
        'offline': offline,
        'resume_from': [],
        'skip_nonexistent': False,

    })

    # Patch every imported function
    mocker.patch.object(
        esmvalcore._config,
        'read_config_user_file',
        create_autospec=True,
        return_value=cfg,
    )
    mocker.patch.object(
        esmvalcore._config,
        'configure_logging',
        create_autospec=True,
    )
    mocker.patch.object(
        esmvalcore._config,
        'DIAGNOSTICS',
        create_autospec=True,
    )
    mocker.patch.object(
        esmvalcore._task,
        'resource_usage_logger',
        create_autospec=True,
    )
    mocker.patch.object(
        esmvalcore.esgf._logon,
        'logon',
        create_autospec=True,
    )
    mocker.patch.object(
        esmvalcore._main,
        'process_recipe',
        create_autospec=True,
    )

    ESMValTool().run(str(recipe), offline=cmd_offline)

    # Check that configuration has been updated from the command line
    assert cfg == reference

    # Check that the correct functions have been called
    esmvalcore._config.read_config_user_file.assert_called_once_with(
        None,
        recipe.stem,
        {},
    )
    esmvalcore._config.configure_logging.assert_called_once_with(
        output_dir=cfg['run_dir'],
        console_log_level=cfg['log_level'],
    )

    if offline:
        esmvalcore.esgf._logon.logon.assert_not_called()
    else:
        esmvalcore.esgf._logon.logon.assert_called_once()

    esmvalcore._task.resource_usage_logger.assert_called_once_with(
        pid=os.getpid(),
        filename=os.path.join(cfg['run_dir'], 'resource_usage.txt'),
    )
    esmvalcore._main.process_recipe.assert_called_once_with(
        recipe_file=recipe,
        config_user=cfg,
    )


@mock.patch('esmvalcore._main.iter_entry_points')
def test_header(mock_entry_points, caplog):

    entry_point = mock.Mock()
    entry_point.dist.project_name = 'MyEntry'
    entry_point.dist.version = 'v42.42.42'
    entry_point.name = 'Entry name'
    mock_entry_points.return_value = [entry_point]
    with caplog.at_level(logging.INFO):
        ESMValTool()._log_header(
            'path_to_config_file',
            ['path_to_log_file1', 'path_to_log_file2']
        )

    assert len(caplog.messages) == 8
    assert caplog.messages[0] == HEADER
    assert caplog.messages[1] == 'Package versions'
    assert caplog.messages[2] == '----------------'
    assert caplog.messages[3] == f'ESMValCore: {__version__}'
    assert caplog.messages[4] == 'MyEntry: v42.42.42'
    assert caplog.messages[5] == '----------------'
    assert caplog.messages[6] == 'Using config file path_to_config_file'
    assert caplog.messages[7] == (
        'Writing program log files to:\n'
        'path_to_log_file1\n'
        'path_to_log_file2'
    )


@mock.patch('os.path.isfile')
def test_get_recipe(is_file):
    """Test get recipe."""
    is_file.return_value = True
    recipe = ESMValTool()._get_recipe('/recipe.yaml')
    assert recipe == pathlib.Path('/recipe.yaml')


@mock.patch('os.path.isfile')
@mock.patch('esmvalcore._config.DIAGNOSTICS')
def test_get_installed_recipe(diagnostics, is_file):
    def encountered(path):
        return path == '/install_folder/recipe.yaml'
    is_file.side_effect = encountered
    diagnostics.recipes = pathlib.Path('/install_folder')
    recipe = ESMValTool()._get_recipe('recipe.yaml')
    assert recipe == pathlib.Path('/install_folder/recipe.yaml')


@mock.patch('os.path.isfile')
def test_get_recipe_not_found(is_file):
    """Test get recipe."""
    is_file.return_value = False
    recipe = ESMValTool()._get_recipe('/recipe.yaml')
    assert recipe == pathlib.Path('/recipe.yaml')
