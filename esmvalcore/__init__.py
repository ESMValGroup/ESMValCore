"""ESMValTool core package."""

import os

from loguru import logger

from ._version import __version__

# Do not show log messages from this package unless the application enables
# them, see https://loguru.readthedocs.io/en/stable/resources/recipes.html#configuring-loguru-to-be-used-by-a-library-or-an-application
# The ``esmvaltool`` command enables them in
# :func:`esmvalcore.config._logging.configure_logging`.
logger.disable("esmvalcore")

__all__ = [
    "__version__",
    "cmor",
    "preprocessor",
]


def get_script_root():
    """Return the location of the ESMValTool installation."""
    return os.path.abspath(os.path.dirname(__file__))
