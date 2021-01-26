"""Fixes for CIESM model."""
from ..common import ClFixHybridPressureCoord


class Cl(ClFixHybridPressureCoord):
    """Fixes for cl."""

    def fix_data(self, cube):
        """
        Fix data.

        Fixes discrepancy between declared units and real units.

        Parameters
        ----------
        cube: iris.cube.Cube
            Input cube.

        Returns
        -------
        iris.cube.Cube

        """
        metadata = cube.metadata
        cube *= 100
        cube.metadata = metadata
        return cube
