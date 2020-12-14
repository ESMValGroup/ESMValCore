"""ESMValCore experimental API module."""

from ._warnings import warnings

warnings.warn(
    '\n  Thank you for trying out the new ESMValCore API.'
    '\n  Note that this API is experimental and may be subject to change.'
    '\n  More info: https://github.com/ESMValGroup/ESMValCore/issues/498', )

from .config import CFG  # noqa: E402

__all__ = [
    'CFG',
]
