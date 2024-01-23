"""Configuration module.

.. data:: CFG

    ESMValCore configuration.

    By default, this will be loaded from the file
    ``~/.esmvaltool/config-user.yml``. If an environment file
    ``ESMVALTOOL_USER_CONFIG`` is set, load the configuration from that path
    instead. If a relative path is given, search in ``~/.esmvaltool``.
"""

from ._config_object import CFG, Config, Session

__all__ = (
    'CFG',
    'Config',
    'Session',
)
