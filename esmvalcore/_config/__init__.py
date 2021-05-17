"""ESMValTool configuration."""
from ._config import (
    get_activity,
    get_institutes,
    get_project_config,
    get_variable_details,
    load_config_developer,
    read_config_developer_file,
    read_config_user_file,
)
from ._diagnostics import DIAGNOSTICS, TAGS
from ._logging import configure_logging

__all__ = (
    'read_config_user_file',
    'read_config_developer_file',
    'load_config_developer',
    'get_variable_details',
    'get_project_config',
    'get_institutes',
    'get_activity',
    'DIAGNOSTICS',
    'TAGS',
    'configure_logging',
)
