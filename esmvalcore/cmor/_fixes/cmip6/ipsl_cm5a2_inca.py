"""Fixes for IPSL-CM5A2-INCA model."""

import iris.coords

from esmvalcore.cmor._fixes.fix import Fix

from .ipsl_cm6a_lr import AllVars as BaseAllVars
from .ipsl_cm6a_lr import Clcalipso as BaseClcalipso
from .ipsl_cm6a_lr import Omon as BaseOmon

AllVars = BaseAllVars


Clcalipso = BaseClcalipso


Omon = BaseOmon


class Snw(Fix):
    """Fixes for ``snw``."""

    def fix_metadata(self, cubes):
        """Fix ``lon`` coordinate.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes

        Returns
        -------
        iris.cube.CubeList

        """

        for cube in cubes:
            for lon_coord in cube.coords(standard_name="longitude"):
                if lon_coord.points[0] == 360.0:
                    lon_coord.points[0] = 0.0

                dim = iris.coords.DimCoord(
                    lon_coord.points,
                    bounds=lon_coord.bounds,
                    standard_name=lon_coord.standard_name,
                    units=lon_coord.units,
                )

                dimension = cube.coord_dims(lon_coord)
                cube.remove_coord(lon_coord)
                cube.add_dim_coord(dim, dimension)

        return cubes
