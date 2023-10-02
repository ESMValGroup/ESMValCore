import warnings
from pathlib import Path

import pytest

import esmvalcore
from esmvalcore.config import CFG, Config
from esmvalcore.exceptions import ESMValCoreDeprecationWarning


def test_no_deprecation_default_cfg():
    """Test that default config does not raise any deprecation warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter('error', category=ESMValCoreDeprecationWarning)
        CFG.reload()
        CFG.start_session('my_session')


def test_no_deprecation_user_cfg():
    """Test that user config does not raise any deprecation warnings."""
    config_file = Path(esmvalcore.__file__).parent / 'config-user.yml'
    with warnings.catch_warnings():
        warnings.simplefilter('error', category=ESMValCoreDeprecationWarning)
        cfg = Config(CFG.copy())
        cfg.load_from_file(config_file)
        cfg.start_session('my_session')


def test_offline_default_cfg():
    """Test that ``offline`` is added for backwards-compatibility."""
    assert CFG['search_esgf'] == 'never'
    assert CFG['offline'] is True


def test_offline_user_cfg():
    """Test that ``offline`` is added for backwards-compatibility."""
    config_file = Path(esmvalcore.__file__).parent / 'config-user.yml'
    cfg = Config(CFG.copy())
    cfg.load_from_file(config_file)
    assert cfg['search_esgf'] == 'never'
    assert cfg['offline'] is True


def test_offline_default_session():
    """Test that ``offline`` is added for backwards-compatibility."""
    session = CFG.start_session('my_session')
    assert session['search_esgf'] == 'never'
    assert session['offline'] is True


def test_offline_user_session():
    """Test that ``offline`` is added for backwards-compatibility."""
    config_file = Path(esmvalcore.__file__).parent / 'config-user.yml'
    cfg = Config(CFG.copy())
    cfg.load_from_file(config_file)
    session = cfg.start_session('my_session')
    assert session['search_esgf'] == 'never'
    assert session['offline'] is True


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
    assert session['search_esgf'] == 'when_missing'


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
    assert CFG['search_esgf'] == 'when_missing'
