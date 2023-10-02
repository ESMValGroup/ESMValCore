import warnings
from pathlib import Path

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
