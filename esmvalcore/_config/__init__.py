import warnings

from ..config import CFG
from ..config._logging import configure_logging
from ..exceptions import ESMValCoreDeprecationWarning

__all__ = [
    'CFG',
    'configure_logging',
    'read_config_user_file',
]

warnings.warn(
    "The private module `esmvalcore._config` has been deprecated in "
    "ESMValCore version 2.8.0 and is scheduled for removal in version 2.9.0. "
    "Please use the public module `esmvalcore.config` instead.",
    ESMValCoreDeprecationWarning,
)


def read_config_user_file(config_file, folder_name, options=None):
    """Read config user file and store settings in a dictionary."""
    CFG.load_from_file(config_file)
    session = CFG.start_session(folder_name)
    session.update(options)
    cfg = session.to_config_user()
    return cfg
