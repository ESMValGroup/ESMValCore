"""ESMValTool core package."""

import os

from ._version import __version__

__all__ = [
    "__version__",
    "cmor",
    "preprocessor",
]


def get_script_root():
    """Return the location of the ESMValTool installation."""
    return os.path.abspath(os.path.dirname(__file__))
