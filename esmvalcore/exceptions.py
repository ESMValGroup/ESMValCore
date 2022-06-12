"""Exceptions that may be raised by ESMValCore."""


class Error(Exception):
    """Base class from which other exceptions are derived."""


class CMORError(Error):
    """An error occurred related to the CMOR module."""


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


class ESMValCoreDeprecationWarning(UserWarning):
    """Custom deprecation warning."""
