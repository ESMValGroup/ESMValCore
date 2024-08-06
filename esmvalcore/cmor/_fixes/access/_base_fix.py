"""Fix base classes for ACCESS-ESM on-the-fly CMORizer."""

import logging

from iris.cube import CubeList

from esmvalcore.cmor._fixes.native_datasets import NativeDatasetFix

logger = logging.getLogger(__name__)


class AccessFix(NativeDatasetFix):
    """Fixes functions."""

    def fix_coord_system(self, cube):
        """Delete coord_system to make CubeList able to merge."""
        for dim in cube.dim_coords:
            if dim.coord_system is not None:
                cube.coord(dim.standard_name).coord_system = None

    def get_cubes_from_multivar(self, cubes):
        """Get cube before calculate from multiple variables."""
        name_list = self.extra_facets.get('raw_name',
                                          self.vardef.short_name)

        data_list = []
        for name in name_list:
            data_list.append(self.get_cube(cubes, name))
        return CubeList(data_list)
