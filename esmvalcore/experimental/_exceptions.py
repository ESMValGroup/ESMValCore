"""ESMValCore exceptions."""

import sys


class SuppressedError(Exception):
    """Errors subclassed from SuppressedError hide the full traceback.

    This can be used for simple user-facing errors that do not need the
    full traceback.
    """


def _suppressed_hook(error, message, traceback):
    """https://stackoverflow.com/a/27674608."""
    if issubclass(error, SuppressedError):
        # Print only the message and hide the traceback
        print(f'{error.__name__}: {message}'.format(error.__name__, message))
    else:
        # Print full traceback
        sys.__excepthook__(error, message, traceback)


sys.excepthook = _suppressed_hook
