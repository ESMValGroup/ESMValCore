"""Define the ESMValCore version."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ESMValCore")
except PackageNotFoundError as exc:
    raise PackageNotFoundError(
        "ESMValCore package not found, please run `pip install -e .` before "
        "importing the package.") from exc
