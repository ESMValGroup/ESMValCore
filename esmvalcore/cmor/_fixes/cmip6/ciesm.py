"""Fixes for CIESM model."""
from ..fix import Fix


class cl(Fix):
    """Fixes for pctisccp."""

    def fix_data(self, cube):
        """
        Fix data.

        Fixes discrepancy between declared units and real units

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
