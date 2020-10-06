"""ESMValTool diagnostics utils."""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def find_diagnostics():
    """Try to find installed diagnostic scripts."""
    try:
        import esmvaltool
    except ImportError:
        return Path.cwd()
    # avoid a crash when there is a directory called
    # 'esmvaltool' that is not a Python package
    if esmvaltool.__file__ is None:
        return Path.cwd()
    return Path(esmvaltool.__file__).absolute().parent


DIAGNOSTICS_PATH = find_diagnostics()
