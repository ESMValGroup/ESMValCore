"""Configuration module.

.. data:: CFG

    ESMValCore configuration.

    By default this will be loaded from the file
    ``~/.esmvaltool/config-user.yml``.
"""
import warnings

from esmvalcore.config import CFG, Config, Session
from esmvalcore.exceptions import ESMValCoreDeprecationWarning

warnings.warn(
    "The module `esmvalcore.experimental.config` has been deprecated in "
    "ESMValCore version 2.8.0 and is scheduled for removal in version 2.9.0. "
    "Please use the module `esmvalcore.config` instead.",
    ESMValCoreDeprecationWarning,
)

__all__ = [
    'CFG',
    'Config',
    'Session',
]
