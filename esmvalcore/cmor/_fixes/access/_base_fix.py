"""Fix base classes for ACCESS-ESM on-the-fly CMORizer."""

import logging

import numpy as np

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
    
    def fix_ocean_dim_coords(self, cube):
        """Fix dim coords of ocean variables"""
        cube.dim_coords[-2].points = np.array([int(i) for i in range(300)])
        cube.dim_coords[-2].standard_name = None
        cube.dim_coords[-2].var_name = 'j'
        cube.dim_coords[-2].long_name = 'cell index along second dimension'
        cube.dim_coords[-2].attributes = None

        cube.dim_coords[-1].points = np.array([int(i) for i in range(360)])
        cube.dim_coords[-1].standard_name = None
        cube.dim_coords[-1].var_name = 'i'
        cube.dim_coords[-1].long_name = 'cell index along first dimension'
        cube.dim_coords[-1].attributes = None
    
    def fix_ocean_aux_coords(self, cube):
        """Fix aux coords of ocean variables"""
        temp_points=[]
        for i in cube.aux_coords[-1].points:
            temp_points.append([j + 360 for j in i if j < 0]+[j for j in i if j >= 0])
        cube.aux_coords[-1].points = np.array(temp_points)
        cube.aux_coords[-1].standard_name = 'longitude'
        cube.aux_coords[-1].long_name = 'longitude'
        cube.aux_coords[-1].var_name = 'longitude'
        cube.aux_coords[-1].attributes = None

        temp_points=[]
        for i in cube.aux_coords[-2].points:
            temp_points.append([j.astype(np.float64) for j in i])
        cube.aux_coords[-2].points = np.array(temp_points)
        cube.aux_coords[-2].standard_name = 'latitude'
        cube.aux_coords[-2].long_name = 'latitude'
        cube.aux_coords[-2].var_name = 'latitude'
        cube.aux_coords[-2].attributes = None
