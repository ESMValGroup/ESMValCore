"""Fixes for CNRM-CM6-1 model."""

from ..fix import Fix
from ..shared import add_scalar_height_coord


class Clcalipso(Fix):
    """Fixes for clcalipso."""

    def fix_metadata(self, cubes):
        """
        Corrects alt40 coordinate standard_name.

        Parameters
        ----------
        cube: iris.cube.CubeList

        Returns
        -------
        iris.cube.CubeList

        """
        for cube in cubes:
            alt40 = cube.coord('alt40')
            alt40.standard_name = 'altitude'

        return cubes

