import logging
import ast

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

    def fix_height_value(self, cube):
        """Fix height value to make it comparable to other dataset."""
        if cube.coord('height').points[0] != 2:
            cube.coord('height').points = [2]

    def get_cubes_from_multivar(self, cubes):
        """Get cube before calculate from multiple variables."""
        rawname_list = self.extra_facets.get('raw_name',
                                             self.vardef.short_name)
        # calculate = self.extra_facets.get('calculate', self.vardef.short_name)
        data_list = []
        for rawname in rawname_list:
            data_list.append(self.get_cube(cubes, rawname))
        return CubeList(data_list)
    
    def fix_rlus_data(self, cubes):
        return cubes[0]-cubes[1]+cubes[2]-cubes[3]
    
    def fix_rsus_data(self, cubes):
        return cubes[0]-cubes[1]
    
    def fix_prc_data(self, cubes):
        return cubes[0]+cubes[1]
    
    def fix_prw_data(self, cubes):
        return cubes[0]-(cubes[1]+cubes[2]+cubes[3])
    
    def fic_rtmt_data(self, cubes):
        return cubes[0]-cubes[1]-cubes[2]