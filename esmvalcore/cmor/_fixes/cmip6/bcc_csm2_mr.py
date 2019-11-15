"""Fixes for BCC-CSM2-MR model."""
from ..fix import Fix


class AllVars(Fix):
    """Fixes for thetao."""

    def fix_metadata(self, cubes):
        """
        Fix cell_area coordinate.

        Parameters
        ----------
        cubes: iris CubeList
            List of cubes to fix

        Returns
        -------
        iris.cube.CubeList

        """
        for cube in cubes:
            cube.coord('latitude').var_name = 'lat'
            cube.coord('longitude').var_name = 'lon'
        return cubes
