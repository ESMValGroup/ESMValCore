import warnings
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ESMValCore")
except PackageNotFoundError:
    # package is not installed
    warnings.warn(
        "No version information available, please install the package.")
    __version__ = "unknown"
