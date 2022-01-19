"""ESMValTool warnings."""

import warnings


def _warning_formatter(message, category, filename, lineno, line=None):
    """Patch warning formatting to not mention itself."""
    return f'{filename}:{lineno}: {category.__name__}: {message}\n'


warnings.formatwarning = _warning_formatter

warnings.warn(
    '\n  Thank you for trying out the new ESMValCore API.'
    '\n  Note that this API is experimental and may be subject to change.'
    '\n  More info: https://github.com/ESMValGroup/ESMValCore/issues/498', )
