"""Fixes for CanESM2 model."""
from ..common import ClFixHybridPressureCoord
from ..fix import Fix


Cl = ClFixHybridPressureCoord


class FgCo2(Fix):
    """Fixes for fgco2."""

    def fix_data(self, cube):
        """
        Fix data.

        Fixes discrepancy between declared units and real units

        Parameters
        ----------
        cube: iris.cube.Cube
            Input cube to fix.

        Returns
        -------
        iris.cube.Cube

        """
        metadata = cube.metadata
        cube *= 12.0 / 44.0
        cube.metadata = metadata
        return cube
