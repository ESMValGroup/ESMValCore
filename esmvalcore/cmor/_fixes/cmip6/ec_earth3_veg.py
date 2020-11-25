"""Fixes for EC-Earth3-Veg model."""
from ..fix import Fix


class Siconca(Fix):
    """Fixes for siconca."""

    def fix_data(self, cube):
        """Fix data.

        Fixes discrepancy between declared units and real units

        Parameters
        ----------
        cube: iris.cube.Cube
            Input cube which needs to be fixed.

        Returns
        -------
        iris.cube.Cube

        """
        cube.data = cube.core_data() * 100.
        return cube
