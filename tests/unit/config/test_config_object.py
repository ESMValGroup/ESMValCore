import os
from collections.abc import MutableMapping
from pathlib import Path

import pytest

import esmvalcore
import esmvalcore.config._config_object
from esmvalcore.config import Config, Session
from esmvalcore.exceptions import (
    ESMValCoreDeprecationWarning,
    InvalidConfigParameter,
)
from tests.integration.test_main import arguments


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


# TODO: remove in v2.14.0
def test_load_from_file(monkeypatch):
    default_config_file = (
        Path(esmvalcore.__file__).parent /
        'config' /
        'config_defaults' /
        'config-user.yml'
    )
    config = Config()
    assert not config
    with pytest.warns(ESMValCoreDeprecationWarning):
        config.load_from_file(default_config_file)
    assert config


# TODO: remove in v2.14.0
def test_load_from_file_filenotfound(monkeypatch):
    """Test `Config.load_from_file`."""
    config = Config()
    assert not config

    expected_path = Path.home() / '.esmvaltool' / 'not_existent_file.yml'
    msg = f"Config file '{expected_path}' does not exist"
    with pytest.raises(FileNotFoundError, match=msg):
        config.load_from_file('not_existent_file.yml')


# TODO: remove in v2.14.0
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


def test_reload(cfg_default, mocker):
    """Test `Config.reload`."""
    path = Path(esmvalcore.__file__).parent / 'config' / 'config_defaults'
    mocker.patch.object(
        esmvalcore.config._config_object,
        'get_config_dirs',
        return_value={'defaults': path},
    )
    cfg = Config()

    cfg.reload()

    assert cfg == cfg_default


def test_reload_fail(mocker, tmp_path):
    """Test `Config.reload`."""
    config_file = tmp_path / 'invalid_config_file.yml'
    config_file.write_text('invalid_option: 1')
    mocker.patch.object(
        esmvalcore.config._config_object,
        'get_config_dirs',
        return_value={'path with invalid config': config_file},
    )
    cfg = Config()

    with pytest.raises(InvalidConfigParameter):
        cfg.reload()


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
        ('esmvaltool', 'run', '--max_parallel_tasks=4'),
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


# TODO: remove in v2.14.0
@pytest.mark.parametrize(
    'filename,env,cli_args,output,env_var_set', TEST_GET_CFG_PATH
)
def test_get_config_user_path(
    filename, env, cli_args, output, env_var_set, monkeypatch, tmp_path
):
    """Test `Config._get_config_user_path`."""
    monkeypatch.delenv('_ESMVALTOOL_USER_CONFIG_FILE_', raising=False)

    # Create empty test file
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'existing_cfg.yml').write_text('')

    if output == 'existing_cfg.yml':
        output = tmp_path / 'existing_cfg.yml'
    else:
        output = Path(output).expanduser()

    if env is not None:
        for (key, val) in env.items():
            monkeypatch.setenv(key, val)
    if cli_args is None:
        cli_args = ['python']

    with arguments(*cli_args):
        config_path = Config._get_config_user_path(filename)
        if env_var_set:
            assert os.environ['_ESMVALTOOL_USER_CONFIG_FILE_'] == str(output)
        else:
            assert '_ESMVALTOOL_USER_CONFIG_FILE_' not in os.environ
    assert isinstance(config_path, Path)
    assert config_path == output


# TODO: remove in v2.14.0
def test_load_user_config_filenotfound():
    """Test `Config._load_user_config`."""
    expected_path = Path.home() / '.esmvaltool' / 'not_existent_file.yml'
    msg = f"Config file '{expected_path}' does not exist"
    with pytest.raises(FileNotFoundError, match=msg):
        Config._load_user_config('not_existent_file.yml')


# TODO: remove in v2.14.0
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


TEST_GET_USER_CONFIG = [
    (
        {},
        ('python',),
        (
            'default user configuration directory',
            Path('~/.config/esmvaltool').expanduser(),
        ),
        False,
    ),
    (
        {},
        ('python', '--config_dir=/config/cli'),
        (
            'default user configuration directory',
            Path('~/.config/esmvaltool').expanduser(),
        ),
        False,
    ),
    (
        {},
        ('esmvaltool', 'run', '--max_parallel_tasks=4'),
        (
            'default user configuration directory',
            Path('~/.config/esmvaltool').expanduser(),
        ),
        True,
    ),
    (
        {},
        ('esmvaltool', '--config_dir'),
        (
            'default user configuration directory',
            Path('~/.config/esmvaltool').expanduser(),
        ),
        True,
    ),
    (
        {},
        ('esmvaltool', 'run', '--config_dir', '/config/cli'),
        ('command line argument', Path('/config/cli')),
        True,
    ),
    (
        {},
        ('esmvaltool', 'run', '--config_dir=/config/cli'),
        ('command line argument', Path('/config/cli')),
        True,
    ),
    (
        {},
        ('esmvaltool', 'run', '--config-dir', '/config/cli'),
        ('command line argument', Path('/config/cli')),
        True,
    ),
    (
        {},
        ('esmvaltool', 'run', '--config-dir=/config/cli'),
        ('command line argument', Path('/config/cli')),
        True,
    ),
    (
        {'_ESMVALTOOL_USER_CONFIG_DIR_': '/config/env'},
        ('esmvaltool', 'run', '--config-dir=/config/cli'),
        (
            '_ESMVALTOOL_USER_CONFIG_DIR_ environment variable',
            Path('/config/env'),
        ),
        True,
    ),
    (
        {'_ESMVALTOOL_USER_CONFIG_DIR_': '/config/env'},
        ('python',),
        (
            '_ESMVALTOOL_USER_CONFIG_DIR_ environment variable',
            Path('/config/env'),
        ),
        True,
    ),
]


@pytest.mark.parametrize(
    'env,cli_args,output,env_var_set', TEST_GET_USER_CONFIG
)
def test_get_user_config(env, cli_args, output, env_var_set, monkeypatch):
    """Test `_get_user_config`."""
    monkeypatch.delenv('_ESMVALTOOL_USER_CONFIG_DIR_', raising=False)

    for (key, val) in env.items():
        monkeypatch.setenv(key, val)

    with arguments(*cli_args):
        config = esmvalcore.config._config_object._get_user_config()
        if env_var_set:
            assert os.environ['_ESMVALTOOL_USER_CONFIG_DIR_'] == str(output[1])
        else:
            assert '_ESMVALTOOL_USER_CONFIG_DIR_' not in os.environ
    assert config == output


def test_get_config_dirs_env(tmp_path, monkeypatch):
    """Test `_get_config_dirs`."""
    monkeypatch.delenv('_ESMVALTOOL_USER_CONFIG_DIR_', raising=False)
    monkeypatch.setenv('ESMVALTOOL_CONFIG_DIR', str(tmp_path))

    config_dirs = esmvalcore.config._config_object.get_config_dirs()

    expected = {
        'defaults':
        Path(esmvalcore.__file__).parent / 'config' / 'config_defaults',
        'default user configuration directory':
        Path('~/.config/esmvaltool').expanduser(),
        'ESMVALTOOL_CONFIG_DIR environment variable': tmp_path,
    }
    assert config_dirs == expected


def test_get_config_dirs_internal_env(tmp_path, monkeypatch):
    """Test `_get_config_dirs`."""
    monkeypatch.setenv('_ESMVALTOOL_USER_CONFIG_DIR_', str(tmp_path))
    monkeypatch.setenv('ESMVALTOOL_CONFIG_DIR', str(tmp_path))

    config_dirs = esmvalcore.config._config_object.get_config_dirs()

    expected = {
        'defaults':
        Path(esmvalcore.__file__).parent / 'config' / 'config_defaults',
        '_ESMVALTOOL_USER_CONFIG_DIR_ environment variable': tmp_path,
        'ESMVALTOOL_CONFIG_DIR environment variable': tmp_path,
    }
    assert config_dirs == expected


def test_get_config_dirs_cli_arg(tmp_path, monkeypatch):
    """Test `_get_config_dirs`."""
    monkeypatch.delenv('_ESMVALTOOL_USER_CONFIG_DIR_', raising=False)
    monkeypatch.delenv('ESMVALTOOL_CONFIG_DIR', raising=False)

    with arguments('esmvaltool', 'run', f'--config_dir={tmp_path}'):
        config_dirs = esmvalcore.config._config_object.get_config_dirs()

    expected = {
        'defaults':
        Path(esmvalcore.__file__).parent / 'config' / 'config_defaults',
        'command line argument': tmp_path,
    }
    assert config_dirs == expected


def test_get_config_dirs_invalid_env(monkeypatch):
    """Test `_get_config_dirs`."""
    monkeypatch.delenv('_ESMVALTOOL_USER_CONFIG_DIR_', raising=False)
    monkeypatch.setenv('ESMVALTOOL_CONFIG_DIR', '/not/a/dir')

    with pytest.raises(NotADirectoryError):
        esmvalcore.config._config_object.get_config_dirs()


def test_get_config_dirs_invalid_cli_arg(monkeypatch):
    """Test `_get_config_dirs`."""
    monkeypatch.delenv('_ESMVALTOOL_USER_CONFIG_DIR_', raising=False)
    monkeypatch.delenv('ESMVALTOOL_CONFIG_DIR', raising=False)

    with pytest.raises(NotADirectoryError):
        with arguments('esmvaltool', 'run', '--config_dir=/not/a/dir'):
            esmvalcore.config._config_object.get_config_dirs()


# TODO: remove in v2.14.0
def test_get_global_config_deprecated(mocker, tmp_path):
    """Test ``get_global_config``."""
    config_file = tmp_path / 'old_config_user.yml'
    config_file.write_text('output_dir: /new/output/dir')
    mocker.patch.object(
        esmvalcore.config._config_object.Config,
        '_get_config_user_path',
        return_value=config_file,
    )
    with pytest.warns(ESMValCoreDeprecationWarning):
        cfg = esmvalcore.config._config_object.get_global_config()

    assert cfg['output_dir'] == Path('/new/output/dir')


# TODO: remove in v2.14.0
def test_get_config_dirs_deprecated(mocker, tmp_path):
    """Test ``get_config_dirs``."""
    config_file = tmp_path / 'old_config_user.yml'
    config_file.write_text('output_dir: /new/output/dir')
    mocker.patch.object(
        esmvalcore.config._config_object.Config,
        '_get_config_user_path',
        return_value=config_file,
    )

    config_dirs = esmvalcore.config._config_object.get_config_dirs()

    expected = {
        'defaults':
        Path(esmvalcore.__file__).parent / 'config' / 'config_defaults',
        'single configuration file [deprecated]': config_file,
    }
    assert config_dirs == expected
