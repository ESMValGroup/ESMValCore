"""Fixes for MIROC ESM CHEM model."""

from esmvalcore.cmor._fixes.fix import Fix


class Tro3(Fix):
    """Fixes for tro3."""

    def fix_data(self, cube):
        """
        Fix data.

        Fixes discrepancy between declared units and real units

        Parameters
        ----------
        cube: iris.cube.Cube

        Returns
        -------
        iris.cube.Cube

        """
        metadata = cube.metadata
        cube *= 1000
        cube.metadata = metadata
        return cube
