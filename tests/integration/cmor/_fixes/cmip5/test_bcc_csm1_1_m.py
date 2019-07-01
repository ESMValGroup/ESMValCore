"""Test Access1-0 fixes."""
import unittest

from iris.cube import Cube
from iris.coords import AuxCoord
import numpy as np

from esmvalcore.cmor._fixes.cmip5.bcc_csm1_1_m import Tos


class TestTos(unittest.TestCase):
    """Test tos fixes."""

    def test_construcot(self):
        """
        Test constructor.

        Minimum test to ensure that at least all fixes are loaded when testing
        """
        Tos()
