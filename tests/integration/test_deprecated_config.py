import warnings
from pathlib import Path

import pytest

import esmvalcore
from esmvalcore._config import read_config_user_file
from esmvalcore.config import CFG
from esmvalcore.config._config_validators import _validators
from esmvalcore.config._validated_config import ValidatedConfig
from esmvalcore.exceptions import ESMValCoreDeprecationWarning


def test_read_config_user():
    config_file = Path(esmvalcore.__file__).parent / 'config-user.yml'
    cfg = read_config_user_file(
        config_file,
        'recipe_test', {'search_esgf': 'default'},
    )
    assert len(cfg) > 1
    assert cfg['search_esgf'] == 'default'


def test_no_deprecation():
    """Test that default config does not raise any deprecation warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter('error', category=ESMValCoreDeprecationWarning)
        CFG.reload()
        CFG.start_session('my_session')


def test_offline_deprecation_validate_config(monkeypatch):
    """Test that the usage of offline is deprecated."""
    msg = "offline"
    monkeypatch.setattr(ValidatedConfig, '_validate', _validators)
    with pytest.warns(ESMValCoreDeprecationWarning, match=msg):
        ValidatedConfig({'offline': True})


def test_offline_deprecation_session_setitem():
    """Test that the usage of offline is deprecated."""
    msg = "offline"
    session = CFG.start_session('my_session')
    session.pop('search_esgf')  # test automatic addition of search_esgf
    with pytest.warns(ESMValCoreDeprecationWarning, match=msg):
        session['offline'] = True
    assert session['offline'] is True
    assert session['search_esgf'] == 'never'


def test_offline_deprecation_session_update():
    """Test that the usage of offline is deprecated."""
    msg = "offline"
    session = CFG.start_session('my_session')
    session.pop('search_esgf')  # test automatic addition of search_esgf
    with pytest.warns(ESMValCoreDeprecationWarning, match=msg):
        session.update({'offline': False})
    assert session['offline'] is False
    assert session['search_esgf'] == 'default'


def test_offline_true_deprecation_config(monkeypatch):
    """Test that the usage of offline is deprecated."""
    msg = "offline"
    monkeypatch.delitem(CFG, 'search_esgf')
    with pytest.warns(ESMValCoreDeprecationWarning, match=msg):
        monkeypatch.setitem(CFG, 'offline', True)
    assert CFG['offline'] is True
    assert CFG['search_esgf'] == 'never'


def test_offline_false_deprecation_config(monkeypatch):
    """Test that the usage of offline is deprecated."""
    msg = "offline"
    monkeypatch.delitem(CFG, 'search_esgf')
    with pytest.warns(ESMValCoreDeprecationWarning, match=msg):
        monkeypatch.setitem(CFG, 'offline', False)
    assert CFG['offline'] is False
    assert CFG['search_esgf'] == 'default'
