"""On-the-fly CMORizer for ACCESS-ESM.

Note
----
This is the first version of ACCESS-ESM CMORizer in for ESMValCore
Currently, only two variables (`tas`,`pr`) is fully supported.
"""
import logging

from iris.cube import CubeList

from esmvalcore.cmor._fixes.native_datasets import NativeDatasetFix

logger = logging.getLogger(__name__)


class AllVars(NativeDatasetFix):

    def fix_coord_system(self, cube):
        """Delete coord_system to make CubeList able to merge."""
        for dim in cube.dim_coords:
            if dim.coord_system is not None:
                cube.coord(dim.standard_name).coord_system = None

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
        cube = self.get_cube(cubes)

        # Fix scalar coordinates (you don't have this at the moment, but this might be helpful as well)
        self.fix_scalar_coords(cube)

        # Fix metadata of variable
        self.fix_var_metadata(cube)

        if 'height_0' in [var.var_name for var in cube.coords()]:
            self.fix_height_metadata(cube)

        # Fix coord system
        self.fix_coord_system(cube)

        return CubeList([cube])
