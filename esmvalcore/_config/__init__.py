"""ESMValTool configuration."""
from ._config import (
    TASKSEP,
    get_activity,
    get_extra_facets,
    get_institutes,
    get_project_config,
    load_config_developer,
)
from ._config_object import CFG, Config, Session
from ._diagnostics import DIAGNOSTICS, TAGS
from ._logging import configure_logging

__all__ = (
    'CFG',
    'DIAGNOSTICS',
    'TAGS',
    'TASKSEP',
    'Config',
    'Session',
    'configure_logging',
    'get_activity',
    'get_extra_facets',
    'get_institutes',
    'get_project_config',
    'load_config_developer',
)
