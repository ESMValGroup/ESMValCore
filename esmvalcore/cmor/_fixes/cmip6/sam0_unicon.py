"""Fixes for SAM0-UNICON model."""
from ..common import ClFixHybridPressureCoord
from ..fix import Fix


Cl = ClFixHybridPressureCoord


Cli = ClFixHybridPressureCoord


Clw = ClFixHybridPressureCoord


class Nbp(Fix):
    """Fixes for nbp."""

    def fix_data(self, cube):
        """Fix data.

        Fixes wrong sign for land surface flux. Tested for v20190323.

        Parameters
        ----------
        cube : iris.cube.Cube
            Input cube.

        Returns
        -------
        iris.cube.Cube

        """
        metadata = cube.metadata
        cube *= -1
        cube.metadata = metadata
        return cube
