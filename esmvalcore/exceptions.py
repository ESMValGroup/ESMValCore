"""Exceptions that may be raised by ESMValCore."""
import logging
import sys

logger = logging.getLogger(__name__)


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
        super().__init__(self)
        self.message = msg
        self.failed_tasks = []

    def __str__(self):
        """Return message string."""
        return self.message


class InputFilesNotFound(RecipeError):
    """Files that are required to run the recipe have not been found."""


ESMVALCORE_DEPRECATION_WARNINGS = set()
"""Set which stores all raised ESMValCoreDeprecationWarnings.

Store all raised ESMValCoreDeprecationWarnings to be able to show them at the
very end of an ESMValTool run (increases visibility for users).
"""


class ESMValCoreDeprecationWarning(UserWarning):
    """Custom deprecation warning."""

    def __init__(self, *args: object) -> None:
        """Store warnings in global variable."""
        super().__init__(*args)
        for msg in args:
            ESMVALCORE_DEPRECATION_WARNINGS.add(msg)


def show_esmvalcore_deprecation_warnings():
    """Show all stored ESMValCore deprecation warnings."""
    if ESMVALCORE_DEPRECATION_WARNINGS:
        logger.warning(
            "Please consider the following ESMValCore deprecation messages:"
        )
    sorted_messages = sorted(ESMVALCORE_DEPRECATION_WARNINGS)
    for msg in sorted_messages:
        logger.warning(msg)
