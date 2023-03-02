from collections.abc import MutableMapping
from pathlib import Path

import pytest

from esmvalcore.config import Config, Session, _config_object
from esmvalcore.exceptions import InvalidConfigParameter


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
    default_config_user_file = Path.home() / '.esmvaltool' / 'config-user.yml'
    assert _config_object.USER_CONFIG == default_config_user_file
    monkeypatch.setattr(
        _config_object,
        'USER_CONFIG',
        _config_object.DEFAULT_CONFIG,
    )
    config = Config()
    assert not config
    config.load_from_file()
    assert config


def test_config_key_error():
    config = Config()
    with pytest.raises(KeyError):
        config['invalid_key']


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
