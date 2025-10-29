"""Fixes for E3SM-1-1 model."""

from iris.cube import Cube

from esmvalcore.cmor.fix import Fix
from esmvalcore.preprocessor._shared import get_array_module


def _mask_greater(cube: Cube, value: float) -> Cube:
    """Mask all data of cube which is greater than ``value``."""
    npx = get_array_module(cube.core_data())
    cube.data = npx.ma.masked_greater(cube.core_data(), value)
    return cube


class Hus(Fix):
    """Fixes for ``hus``."""

    def fix_data(self, cube: Cube) -> Cube:
        """Fix data.

        Fix values that are not properly masked.

        Parameters
        ----------
        cube: iris.cube.Cube
            Input cube.

        Returns
        -------
        iris.cube.Cube

        """
        return _mask_greater(cube, 1000.0)


Ta = Hus


Ua = Hus


Va = Hus
