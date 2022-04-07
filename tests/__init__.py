"""Provides testing capabilities for :mod:`esmvaltool` package."""
import unittest
from unittest import mock

import numpy as np

from esmvalcore.preprocessor import PreprocessorFile as PreprocessorFileBase


def assert_array_equal(a, b):
    """Assert that array a equals array b."""
    np.testing.assert_array_equal(a, b)
    if np.ma.isMaskedArray(a) or np.ma.isMaskedArray(b):
        np.testing.assert_array_equal(a.mask, b.mask)


class Test(unittest.TestCase):
    """Provides esmvaltool specific testing functionality."""
    def _remove_testcase_patches(self):
        """
        Helper method to remove per-testcase patches installed by
        :meth:`patch`.

        """
        # Remove all patches made, ignoring errors.
        for patch in self.testcase_patches:
            patch.stop()

        # Reset per-test patch control variable.
        self.testcase_patches.clear()

    def patch(self, *args, **kwargs):
        """Install a patch to be removed automatically after the current test.

        The patch is created with :func:`unittest.mock.patch`.

        Parameters
        ----------
        args : list
            The parameters to be passed to :func:`unittest.mock.patch`.
        kwargs : dict
            The keyword parameters to be passed to :func:`unittest.mock.patch`.

        Returns
        -------
            The substitute mock instance returned by
            :func:`unittest.patch.start`.
        """
        # Make the new patch and start it.
        patch = unittest.mock.patch(*args, **kwargs)
        start_result = patch.start()

        # Create the per-testcases control variable if it does not exist.
        # NOTE: this mimics a setUp method, but continues to work when a
        # subclass defines its own setUp.
        if not hasattr(self, 'testcase_patches'):
            self.testcase_patches = {}

        # When installing the first patch, schedule remove-all at cleanup.
        if not self.testcase_patches:
            self.addCleanup(self._remove_testcase_patches)

        # Record the new patch and start object for reference.
        self.testcase_patches[patch] = start_result

        # Return patch replacement object.
        return start_result

    def assert_array_equal(self, a, b):
        assert_array_equal(a, b)


class PreprocessorFile(mock.Mock):
    """Mocked PreprocessorFile."""

    def __init__(self, cubes, filename, attributes, settings=None, **kwargs):
        """Initialize with cubes."""
        super().__init__(spec=PreprocessorFileBase, **kwargs)
        self.cubes = cubes
        self.filename = filename
        self.attributes = attributes
        if settings is None:
            self.settings = {}
        else:
            self.settings = settings
        self.mock_ancestors = set()
        self.wasderivedfrom = mock.Mock(side_effect=self.mock_ancestors.add)

    group = PreprocessorFileBase.group
