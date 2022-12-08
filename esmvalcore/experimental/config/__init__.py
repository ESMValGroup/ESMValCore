"""ESMValTool config module.

.. data:: CFG

    ESMValCore configuration.
    By default this will loaded from the file ~/.esmvaltool/config-user.yml.

"""

from ._config_object import CFG, Config, Session

__all__ = [
    'CFG',
    'Config',
    'Session',
]
