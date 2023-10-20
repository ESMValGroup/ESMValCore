"""Fixes for CNRM-ESM2-1 model."""
from ..fix import Fix


class Dpn2o(Fix):
    """Fixes for Dpn2o."""

    def fix_metadata(self, cubes):
        """Fix metadata.

        Changes unit from ppm to natm.

        Parameters
        ----------
        cubes: iris.cube.CubeList
            Input cubes to fix.

        Returns
        -------
        iris.cube.CubeList

        """
        for cube in cubes:
            cube.units = 'natm'
        return cubes

    def fix_data(self, cube):
        """
        Fix data.

        Adjust data to reflect change of unit.

        Parameters
        ----------
        cube: iris.cube.Cube
            Input cube to fix.

        Returns
        -------
        iris.cube.Cube

        """
        cube.data *= -1.0
        return cube
