"""Exceptions that may be raised by ESMValCore."""
import sys


class Error(Exception):
    """Base class from which other exceptions are derived."""


class SuppressedError(Exception):
    """Errors subclassed from SuppressedError hide the full traceback.

    This can be used for simple user-facing errors that do not need the
    full traceback.
    """


def _suppressed_hook(error, message, traceback):
    """https://stackoverflow.com/a/27674608."""
    if issubclass(error, SuppressedError):
        # Print only the message and hide the traceback
        print(f'{error.__name__}: {message}', file=sys.stderr)
    else:
        # Print full traceback
        sys.__excepthook__(error, message, traceback)


sys.excepthook = _suppressed_hook


class InvalidConfigParameter(Error, SuppressedError):
    """Config parameter is invalid."""


class MissingConfigParameter(UserWarning):
    """Config parameter is missing."""


class RecipeError(Error):
    """Recipe contains an error."""

    def __init__(self, msg):
        super().__init__(msg)
        self.message = msg
        self.failed_tasks = []


class InputFilesNotFound(RecipeError):
    """Files that are required to run the recipe have not been found."""


class ESMValCoreDeprecationWarning(UserWarning):
    """Custom deprecation warning."""
