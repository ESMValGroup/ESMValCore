"""On-the-fly CMORizer for ACCESS-ESM."""
import logging

from iris.cube import CubeList

from ._base_fix import AccessFix

logger = logging.getLogger(__name__)


class AllVars(AccessFix):
    """Fixes for all variables."""

    def fix_metadata(self, cubes):
        """Fix metadata.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList
        """
        if len(cubes) == 1:
            cube = cubes[0]
        else:
            cube = self.get_cube(cubes)

        # Fix coordinates
        self.fix_scalar_coords(cube)
        self.fix_var_metadata(cube)
        self.fix_lon_metadata(cube)
        self.fix_lat_metadata(cube)

        # Fix coordinate 'height'
        if 'height_0' in [var.var_name for var in cube.coords()]:
            self.fix_height_metadata(cube)
        # Fix coordinate 'pressure'
        if 'pressure' in [var.var_name for var in cube.coords()]:
            self.fix_plev_metadata(cube, coord='pressure')

        # Fix coord system
        self.fix_coord_system(cube)

        return CubeList([cube])


class Rlus(AccessFix):
    """Fixes for Rlus."""

    def fix_rlus_data(self, cubes):
        """Fix rlus data."""
        return cubes[0] - cubes[1] + cubes[2] - cubes[3]

    def fix_metadata(self, cubes):
        """Fix metadata.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList
        """
        cubes = self.get_cubes_from_multivar(cubes)

        cube = self.fix_rlus_data(cubes)

        return CubeList([cube])


class Rsus(AccessFix):
    """Fixes for Rsus."""

    def fix_rsus_data(self, cubes):
        """Fix rsus data."""
        return cubes[0] - cubes[1]

    def fix_metadata(self, cubes):
        """Fix metadata.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList
        """
        cubes = self.get_cubes_from_multivar(cubes)

        cube = self.fix_rsus_data(cubes)

        return CubeList([cube])


class Tas(AccessFix):
    """Fixes for Rsus."""

    def fix_metadata(self, cubes):
        """Fix metadata.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList
        """
        cube = self.get_cube(cubes)

        self.fix_height_metadata(cube)
        self.fix_height_value(cube)

        return CubeList([cube])

    def fix_height_value(self, cube):
        """Fix height value to make it comparable to other dataset."""
        if cube.coord('height').points[0] != 2:
            cube.coord('height').points = [2]
