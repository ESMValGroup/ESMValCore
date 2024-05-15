import logging

import csv

from iris.cube import CubeList

from ..native_datasets import NativeDatasetFix

from esmvalcore.cmor.check import cmor_check

logger = logging.getLogger(__name__)

class tas(NativeDatasetFix):

    def __init__(self,vardef,
        extra_facets,
        session,
        frequency):

        super().__init__(vardef,extra_facets,session,frequency)

        self.cube=None

    def fix_height2m(self,cube,cubes):
        if cube.coords('height'):
            # In case a scalar height is required, remove it here (it is added
            # at a later stage). The step _fix_height() is designed to fix
            # non-scalar height coordinates.
            if (cube.coord('height').shape[0] == 1 and (
                    'height2m' in self.vardef.dimensions or
                    'height10m' in self.vardef.dimensions)):
                # If height is a dimensional coordinate with length 1, squeeze
                # the cube.
                # Note: iris.util.squeeze is not used here since it might
                # accidentally squeeze other dimensions.
                if cube.coords('height', dim_coords=True):
                    slices = [slice(None)] * cube.ndim
                    slices[cube.coord_dims('height')[0]] = 0
                    cube = cube[tuple(slices)]
                cube.remove_coord('height')
            else:
                cube = self._fix_height(cube, cubes)
            return cube
        
    def fix_height_name(self):
        if self.cube.coord('height').var_name!='height':
            self.cube.coord('height').var_name='height'
    
    def fix_long_name(self):
        self.cube.long_name ='Near-Surface Air Temperature'

    def fix_var_name(self):
        self.cube.var_name='tas'

    def fix_metadata(self, cubes):

        master_map_path='./master_map.csv'

        with open (master_map_path,'r') as map:
            reader=csv.reader(map, delimiter=',')
            for raw in reader:
                if raw[0]=='tas':
                    tas_map=raw
                    break

        # original_short_name='air_temperature'
        original_short_name='fld_s03i236'

        self.cube= self.get_cube(cubes, var_name=original_short_name)

        self.fix_height_name()

        self.fix_long_name()

        self.fix_var_name()

        cube_checked= cmor_check(cube=self.cube,cmor_table='CMIP6',mip='Amon',short_name='tas',check_level=1)
        
        print('Successfully get the cube(tas)')

        return CubeList([cube_checked])


class pr(NativeDatasetFix):

    def __init__(self,vardef,
        extra_facets,
        session,
        frequency):

        super().__init__(vardef,extra_facets,session,frequency)

        self.cube=None

    def fix_height2m(self,cube,cubes):
        if cube.coords('height'):
            # In case a scalar height is required, remove it here (it is added
            # at a later stage). The step _fix_height() is designed to fix
            # non-scalar height coordinates.
            if (cube.coord('height').shape[0] == 1 and (
                    'height2m' in self.vardef.dimensions or
                    'height10m' in self.vardef.dimensions)):
                # If height is a dimensional coordinate with length 1, squeeze
                # the cube.
                # Note: iris.util.squeeze is not used here since it might
                # accidentally squeeze other dimensions.
                if cube.coords('height', dim_coords=True):
                    slices = [slice(None)] * cube.ndim
                    slices[cube.coord_dims('height')[0]] = 0
                    cube = cube[tuple(slices)]
                cube.remove_coord('height')
            else:
                cube = self._fix_height(cube, cubes)
            return cube
        else:
            return cube
    # def fix_height_name(self, cube):
    #     for coord in cube.dim_coords:
    #         if coord.var_name=='height':
    #             if cube.coord('height').var_name!='height':
    #                 cube.coord('height').var_name='height'
    #     return cube

    def fix_var_name(self):
        self.cube.var_name='pr'
    
    def fix_long_name(self):
        self.cube.long_name ='Precipitation'

    # def fix_coord_system(self,cube):
    #     cube.coords('latitude')[0].coord_system=

    def fix_coord_system(self):
        for dim in self.cube.dim_coords:
            if dim.coord_system!=None:
                self.cube.coord(dim.standard_name).coord_system=None

    def fix_metadata(self, cubes):

        master_map_path='./master_map.csv'

        with open (master_map_path,'r') as map:
            reader=csv.reader(map, delimiter=',')
            for raw in reader:
                if raw[0]=='pr':
                    pr_map=raw
                    break

        # original_short_name='air_temperature'
        original_short_name='fld_s05i216'

        cube= self.get_cube(cubes, var_name=original_short_name)

        self.fix_var_name()

        self.fix_long_name()

        self.fix_coord_system()

        cube_checked= cmor_check(cube=cube,cmor_table='CMIP6',mip='Amon',short_name='pr',check_level=1)

        print('Successfully get the cube(pr)')

        return CubeList([cube_checked])

