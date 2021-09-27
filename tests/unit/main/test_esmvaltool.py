import os

import pytest

import esmvalcore._config
import esmvalcore._main
import esmvalcore._task
import esmvalcore.esgf
from esmvalcore._main import ESMValTool
from esmvalcore.cmor.check import CheckLevels


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
        'run_dir': str(output_dir / 'run_dir'),
        'preproc_dir': str(output_dir / 'preproc_dir'),
        'log_level': 'info',
        'config_file': tmp_path / '.esmvaltool' / 'config-user.yml',
        'offline': cfg_offline,
    }

    # Expected configuration after updating from command line.
    reference = dict(cfg)
    reference.update({
        'offline': offline,
        'skip-nonexistent': False,
        'diagnostics': set(),
        'check_level': CheckLevels.DEFAULT
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
        recipe_file=str(recipe),
        config_user=cfg,
    )
