"""Configuration module.

.. data:: CFG

    ESMValCore configuration.

    By default, this will be loaded from the file
    ``~/.esmvaltool/config-user.yml``. If a relative path is given, search also
    in ``~/.esmvaltool`` (in addition to the current working directory).

"""

from ._config_object import CFG, Config, Session

__all__ = (
    'CFG',
    'Config',
    'Session',
)
