import contextlib
import os
import sys
from collections.abc import MutableMapping
from copy import deepcopy
from pathlib import Path

import pytest

import esmvalcore
import esmvalcore.config._config_object
from esmvalcore.config import Config, Session
from esmvalcore.exceptions import InvalidConfigParameter
from tests.integration.test_main import arguments


@contextlib.contextmanager
def environment(**kwargs):
    """Temporary environment variables."""
    backup = deepcopy(os.environ)
    os.environ = kwargs
    yield
    os.environ = backup


def test_config_class():
    config = {
        'log_level': 'info',
        'exit_on_warning': False,
        'output_file_type': 'png',
        'output_dir': './esmvaltool_output',
        'auxiliary_data_dir': './auxiliary_data',
        'save_intermediary_cubes': False,
        'remove_preproc_dir': True,
        'max_parallel_tasks': None,
        'profile_diagnostic': False,
        'rootpath': {
            'CMIP6': '~/data/CMIP6'
        },
        'drs': {
            'CMIP6': 'default'
        },
    }

    cfg = Config(config)

    assert isinstance(cfg['output_dir'], Path)
    assert isinstance(cfg['auxiliary_data_dir'], Path)

    from esmvalcore.config._config import CFG as CFG_DEV
    assert CFG_DEV


def test_config_update():
    config = Config({'output_dir': 'directory'})
    fail_dict = {'output_dir': 123}

    with pytest.raises(InvalidConfigParameter):
        config.update(fail_dict)


def test_set_bad_item():
    config = Config({'output_dir': 'config'})
    with pytest.raises(InvalidConfigParameter) as err_exc:
        config['bad_item'] = 47

    assert str(err_exc.value) == '`bad_item` is not a valid config parameter.'


def test_config_init():
    config = Config()
    assert isinstance(config, MutableMapping)


def test_load_from_file(monkeypatch):
    default_config_file = Path(esmvalcore.__file__).parent / 'config-user.yml'
    config = Config()
    assert not config
    config.load_from_file(default_config_file)
    assert config


def test_load_from_file_filenotfound(monkeypatch):
    """Test `Config.load_from_file`."""
    config = Config()
    assert not config

    expected_path = Path.home() / '.esmvaltool' / 'not_existent_file.yml'
    msg = f"Config file '{expected_path}' does not exist"
    with pytest.raises(FileNotFoundError, match=msg):
        config.load_from_file('not_existent_file.yml')


def test_load_from_file_invalidconfigparameter(monkeypatch, tmp_path):
    """Test `Config.load_from_file`."""
    monkeypatch.chdir(tmp_path)
    cfg_path = tmp_path / 'test.yml'
    cfg_path.write_text('invalid_param: 42')

    config = Config()
    assert not config

    msg = (
        f"Failed to parse user configuration file {cfg_path}: `invalid_param` "
        f"is not a valid config parameter."
    )
    with pytest.raises(InvalidConfigParameter, match=msg):
        config.load_from_file(cfg_path)


def test_config_key_error():
    config = Config()
    with pytest.raises(KeyError):
        config['invalid_key']


def test_reload():
    """Test `Config.reload`."""
    cfg_path = Path(esmvalcore.__file__).parent / 'config-user.yml'
    config = Config(config_file=cfg_path)
    config.reload()
    assert config['config_file'] == cfg_path


def test_reload_fail():
    """Test `Config.reload`."""
    config = Config()
    msg = (
        "Cannot reload configuration, option 'config_file' is missing; make "
        "sure to only use the `CFG` object from the `esmvalcore.config` module"
    )
    with pytest.raises(ValueError, match=msg):
        config.reload()


def test_session():
    config = Config({'output_dir': 'config'})

    session = config.start_session('recipe_name')
    assert session == config

    session['output_dir'] = 'session'
    assert session != config


def test_session_key_error():
    session = Session({})
    with pytest.raises(KeyError):
        session['invalid_key']


TEST_GET_CFG_PATH = [
    (None, None, None, '~/.esmvaltool/config-user.yml', False),
    (
        None,
        None,
        ('any_other_module', '--config_file=cli.yml'),
        '~/.esmvaltool/config-user.yml',
        False,
    ),
    (
        None,
        None,
        ('esmvaltool', 'run', '--max-parallel-tasks=4'),
        '~/.esmvaltool/config-user.yml',
        True,
    ),
    (
        None,
        None,
        ('esmvaltool', '--config_file'),
        '~/.esmvaltool/config-user.yml',
        True,
    ),
    (
        None,
        None,
        ('esmvaltool', 'run', '--config_file=/cli.yml'),
        '/cli.yml',
        True,
    ),
    (
        None,
        None,
        ('esmvaltool', 'run', '--config_file=/cli.yml'),
        '/cli.yml',
        True,
    ),
    (
        None,
        None,
        ('esmvaltool', 'run', '--config-file', '/cli.yml'),
        '/cli.yml',
        True,
    ),
    (
        None,
        None,
        ('esmvaltool', 'run', '--config-file=/cli.yml'),
        '/cli.yml',
        True,
    ),
    (
        None,
        None,
        ('esmvaltool', 'run', '--config-file=relative_cli.yml'),
        '~/.esmvaltool/relative_cli.yml',
        True,
    ),
    (
        None,
        None,
        ('esmvaltool', 'run', '--config-file=existing_cfg.yml'),
        'existing_cfg.yml',
        True,
    ),
    (
        None,
        {'_ESMVALTOOL_USER_CONFIG_FILE_': '/env.yml'},
        ('esmvaltool', 'run', '--config-file=/cli.yml'),
        '/env.yml',
        True,
    ),
    (
        None,
        {'_ESMVALTOOL_USER_CONFIG_FILE_': '/env.yml'},
        None,
        '/env.yml',
        True,
    ),
    (
        None,
        {'_ESMVALTOOL_USER_CONFIG_FILE_': 'existing_cfg.yml'},
        ('esmvaltool', 'run', '--config-file=/cli.yml'),
        'existing_cfg.yml',
        True,
    ),
    (
        '/filename.yml',
        {'_ESMVALTOOL_USER_CONFIG_FILE_': '/env.yml'},
        ('esmvaltool', 'run', '--config-file=/cli.yml'),
        '/filename.yml',
        True,
    ),
    (
        '/filename.yml',
        None,
        ('esmvaltool', 'run', '--config-file=/cli.yml'),
        '/filename.yml',
        True,
    ),
    ('/filename.yml', None, None, '/filename.yml', False),
    (
        'filename.yml',
        None,
        None,
        '~/.esmvaltool/filename.yml',
        False,
    ),
    (
        'existing_cfg.yml',
        {'_ESMVALTOOL_USER_CONFIG_FILE_': '/env.yml'},
        ('esmvaltool', 'run', '--config-file=/cli.yml'),
        'existing_cfg.yml',
        True,
    ),
]


@pytest.mark.parametrize(
    'filename,env,cli_args,output,env_var_set', TEST_GET_CFG_PATH
)
def test_get_config_user_path(
    filename, env, cli_args, output, env_var_set, monkeypatch, tmp_path
):
    """Test `Config._get_config_user_path`."""
    # Create empty test file
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'existing_cfg.yml').write_text('')

    if env is None:
        env = {}
    if cli_args is None:
        cli_args = sys.argv

    if output == 'existing_cfg.yml':
        output = tmp_path / 'existing_cfg.yml'
    else:
        output = Path(output).expanduser()

    with environment(**env), arguments(*cli_args):
        config_path = Config._get_config_user_path(filename)
        if env_var_set:
            assert os.environ['_ESMVALTOOL_USER_CONFIG_FILE_'] == str(output)
        else:
            assert '_ESMVALTOOL_USER_CONFIG_FILE_' not in os.environ
    assert isinstance(config_path, Path)
    assert config_path == output


def test_load_user_config_filenotfound():
    """Test `Config._load_user_config`."""
    expected_path = Path.home() / '.esmvaltool' / 'not_existent_file.yml'
    msg = f"Config file '{expected_path}' does not exist"
    with pytest.raises(FileNotFoundError, match=msg):
        Config._load_user_config('not_existent_file.yml')


def test_load_user_config_invalidconfigparameter(monkeypatch, tmp_path):
    """Test `Config._load_user_config`."""
    monkeypatch.chdir(tmp_path)
    cfg_path = tmp_path / 'test.yml'
    cfg_path.write_text('invalid_param: 42')

    msg = (
        f"Failed to parse user configuration file {cfg_path}: `invalid_param` "
        f"is not a valid config parameter."
    )
    with pytest.raises(InvalidConfigParameter, match=msg):
        Config._load_user_config(cfg_path)
