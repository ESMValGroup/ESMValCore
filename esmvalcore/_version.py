"""Define the ESMValCore version."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ESMValCore")
except PackageNotFoundError as exc:
    msg = (
        "ESMValCore package not found, please run `pip install -e .` before "
        "importing the package."
    )
    raise PackageNotFoundError(
        msg,
    ) from exc
