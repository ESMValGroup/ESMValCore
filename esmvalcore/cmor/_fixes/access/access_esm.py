"""On-the-fly CMORizer for ACCESS-ESM."""
import logging

from iris.cube import CubeList

from esmvalcore.cmor._fixes.native_datasets import NativeDatasetFix

logger = logging.getLogger(__name__)


class AllVars(NativeDatasetFix):
    """Fixes for all variables."""

    def fix_coord_system(self, cube):
        """Delete coord_system to make CubeList able to merge."""
        for dim in cube.dim_coords:
            if dim.coord_system is not None:
                dim.coord_system = None

    def fix_height_value(self, cube):
        """Fix height value to make it comparable to other dataset."""
        if cube.coord('height').points[0] != 2:
            cube.coord('height').points = [2]

    def calculate_data_from_multivar(self, cubes):
        """Get cube before calculate from multiple variables."""
        rawname_list = self.extra_facets.get('raw_name',
                                             self.vardef.short_name)
        var = []
        for rawname in rawname_list:
            var.append(self.get_cube(cubes, rawname))
        if self.vardef.short_name == 'rsus':
            cube = var[0] - var[1]
        if self.vardef.short_name == 'rlus':
            cube = var[0] - var[1] + var[2] - var[3]
        return cube

    def fix_metadata(self, cubes):
        """Fix metadata.

        Fix name of coordinate(height), long name and variable name of
        variable(tas).

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList
        """
        if isinstance(
                self.extra_facets.get('raw_name', self.vardef.short_name),
                list):
            cube = self.calculate_data_from_multivar(cubes)
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
            self.fix_height_value(cube)
        # Fix coordinate 'pressure'
        if 'pressure' in [var.var_name for var in cube.coords()]:
            self.fix_plev_metadata(cube, coord='pressure')

        # Fix coord system
        self.fix_coord_system(cube)

        return CubeList([cube])
