"""ESMValTool Warnings."""

import warnings


def warning_formatter(message,
                      category,
                      filename,
                      lineno,
                      file=None,
                      line=None):
    """Patch warning formatting to not mention itself."""
    return f'{filename}:{lineno}: {category.__name__}: {message}\n'


warnings.formatwarning = warning_formatter
