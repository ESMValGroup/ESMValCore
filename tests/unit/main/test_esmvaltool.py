import logging
import os
from pathlib import Path
from unittest import mock

import pytest

import esmvalcore._main
import esmvalcore._task
import esmvalcore.config
import esmvalcore.config._logging
import esmvalcore.esgf
from esmvalcore import __version__
from esmvalcore._main import HEADER, ESMValTool
from esmvalcore.exceptions import RecipeError

LOGGER = logging.getLogger(__name__)


@pytest.fixture
def cfg(mocker, tmp_path):
    """Mock `esmvalcore.config.CFG`."""
    session = mocker.MagicMock()

    cfg_dict = {}
    session.__getitem__.side_effect = cfg_dict.__getitem__
    session.__setitem__.side_effect = cfg_dict.__setitem__

    output_dir = tmp_path / 'esmvaltool_output'
    session.session_dir = output_dir / 'recipe_test'
    session.run_dir = session.session_dir / 'run_dir'
    session.preproc_dir = session.session_dir / 'preproc_dir'
    session._fixed_file_dir = session.preproc_dir / 'fixed_files'

    cfg = mocker.Mock()
    cfg.start_session.return_value = session

    return cfg


@pytest.fixture
def session(cfg):
    return cfg.start_session.return_value


@pytest.mark.parametrize('argument,value', [
    ('max_datasets', 2),
    ('max_years', 2),
    ('skip_nonexistent', True),
    ('search_esgf', 'when_missing'),
    ('diagnostics', 'diagnostic_name/group_name'),
    ('check_level', 'strict'),
])
def test_run_command_line_config(mocker, cfg, argument, value):
    """Check that the configuration is updated from the command line."""
    mocker.patch.object(
        esmvalcore.config,
        'CFG',
        cfg,
    )
    session = cfg.start_session.return_value

    program = ESMValTool()
    recipe_file = '/path/to/recipe_test.yml'
    config_file = '/path/to/config-user.yml'

    mocker.patch.object(program, '_get_recipe', return_value=Path(recipe_file))
    mocker.patch.object(program, '_run')

    program.run(recipe_file, config_file, **{argument: value})

    cfg.load_from_file.assert_called_with(config_file)
    cfg.start_session.assert_called_once_with(Path(recipe_file).stem)
    program._get_recipe.assert_called_with(recipe_file)
    program._run.assert_called_with(program._get_recipe.return_value, session)

    assert session[argument] == value


@pytest.mark.parametrize('search_esgf', ['never', 'when_missing', 'always'])
def test_run(mocker, session, search_esgf):
    session['search_esgf'] = search_esgf
    session['log_level'] = 'default'
    session['config_file'] = '/path/to/config-user.yml'
    session['remove_preproc_dir'] = True
    session['save_intermediary_cubes'] = False
    session.cmor_log.read_text.return_value = 'WARNING: attribute not present'

    recipe = Path('/recipe_dir/recipe_test.yml')

    # Patch every imported function
    mocker.patch.object(
        esmvalcore.config._logging,
        'configure_logging',
        create_autospec=True,
    )
    mocker.patch.object(
        esmvalcore.config._diagnostics,
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

    ESMValTool()._run(recipe, session=session)

    # Check that the correct functions have been called
    esmvalcore.config._logging.configure_logging.assert_called_once_with(
        output_dir=session.run_dir,
        console_log_level=session['log_level'],
    )

    if search_esgf == 'never':
        esmvalcore.esgf._logon.logon.assert_not_called()
    else:
        esmvalcore.esgf._logon.logon.assert_called_once()

    esmvalcore._task.resource_usage_logger.assert_called_once_with(
        pid=os.getpid(),
        filename=session.run_dir / 'resource_usage.txt',
    )
    esmvalcore._main.process_recipe.assert_called_once_with(
        recipe_file=recipe,
        session=session,
    )


def test_run_session_dir_exists(session):
    program = ESMValTool()
    session.session_dir.mkdir(parents=True)
    session_dir = session.session_dir
    program._create_session_dir(session)
    assert session.session_name == f"{session_dir.name}-1"


def test_run_session_dir_exists_alternative_fails(mocker, session):
    mocker.patch.object(
        esmvalcore._main.Path,
        'mkdir',
        side_effect=FileExistsError,
    )
    program = ESMValTool()
    with pytest.raises(RecipeError):
        program._create_session_dir(session)


def test_clean_preproc_dir(session):
    session.preproc_dir.mkdir(parents=True)
    session._fixed_file_dir.mkdir(parents=True)
    session['remove_preproc_dir'] = True
    session['save_intermediary_cubes'] = False
    program = ESMValTool()
    program._clean_preproc(session)
    assert not session.preproc_dir.exists()
    assert not session._fixed_file_dir.exists()


def test_do_not_clean_preproc_dir(session):
    session.preproc_dir.mkdir(parents=True)
    session._fixed_file_dir.mkdir(parents=True)
    session['remove_preproc_dir'] = False
    session['save_intermediary_cubes'] = True
    program = ESMValTool()
    program._clean_preproc(session)
    assert session.preproc_dir.exists()
    assert session._fixed_file_dir.exists()


@mock.patch('esmvalcore._main.entry_points')
def test_header(mock_entry_points, caplog):

    entry_point = mock.Mock()
    entry_point.dist.name = 'MyEntry'
    entry_point.dist.version = 'v42.42.42'
    entry_point.name = 'Entry name'
    mock_entry_points.return_value = [entry_point]
    with caplog.at_level(logging.INFO):
        ESMValTool()._log_header(
            'path_to_config_file',
            ['path_to_log_file1', 'path_to_log_file2'],
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
    assert recipe == Path('/recipe.yaml')


@mock.patch('os.path.isfile')
@mock.patch('esmvalcore.config._diagnostics.DIAGNOSTICS')
def test_get_installed_recipe(diagnostics, is_file):

    def encountered(path):
        return Path(path) == Path('/install_folder/recipe.yaml')

    is_file.side_effect = encountered
    diagnostics.recipes = Path('/install_folder')
    recipe = ESMValTool()._get_recipe('recipe.yaml')
    assert recipe == Path('/install_folder/recipe.yaml')


@mock.patch('os.path.isfile')
def test_get_recipe_not_found(is_file):
    """Test get recipe."""
    is_file.return_value = False
    recipe = ESMValTool()._get_recipe('/recipe.yaml')
    assert recipe == Path('/recipe.yaml')
