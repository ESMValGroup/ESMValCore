import warnings

from esmvalcore.config import CFG
from esmvalcore.exceptions import ESMValCoreDeprecationWarning


def test_no_deprecation_default_cfg():
    """Test that default config does not raise any deprecation warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=ESMValCoreDeprecationWarning)
        CFG.reload()
        CFG.start_session("my_session")
