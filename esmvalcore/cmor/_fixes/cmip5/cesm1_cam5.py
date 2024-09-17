"""Fixes for CESM1-CAM5 model."""

from ..fix import Fix


class Cl(Fix):
    """Fixes for cl."""

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
        cube *= 100
        cube.metadata = metadata
        return cube
