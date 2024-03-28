import logging

import csv

from iris.cube import CubeList

from ..native_datasets import NativeDatasetFix

from esmvalcore.cmor.check import cmor_check

logger = logging.getLogger(__name__)

class tas(NativeDatasetFix):

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
    def fix_height_name(self, cube):
        if cube.coord('height').var_name!='height':
            cube.coord('height').var_name='height'
        return cube
    
    def fix_long_name(self, cube):
        cube.long_name ='Near-Surface Air Temperature'
        return cube

    def fix_var_name(self,cube):
        cube.var_name='tas'
        return cube

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

        cube= self.get_cube(cubes, var_name=original_short_name)

        print('Successfully get the cube(tas)')

        # print('self.vardef:',self.vardef.dimensions)

        # print('height shape:',cube.coord('height').shape[0])

        # print(cube)
        # cube=self.fix_height2m(cube,cubes)

        cube = self.fix_height_name(cube)

        cube = self.fix_long_name(cube)

        print('standard_name:',cube.standard_name)

        print('long_name:',cube.long_name)

        cube_checked= cmor_check(cube=cube,cmor_table='CMIP6',mip='Amon',short_name='tas',check_level=1)
        

        return CubeList([cube_checked])


class pr(NativeDatasetFix):

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

    def fix_var_name(self,cube):
        cube.var_name='pr'
        return cube
    
    def fix_long_name(self, cube):
        cube.long_name ='Precipitation'
        return cube

    # def fix_coord_system(self,cube):
    #     cube.coords('latitude')[0].coord_system=

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

        cube=self.fix_var_name(cube)

        cube=self.fix_long_name(cube)

        cube_checked= cmor_check(cube=cube,cmor_table='CMIP6',mip='Amon',short_name='pr',check_level=1)

        print('Successfully get the cube(pr)')

        # print('self.vardef:',self.vardef.dimensions)

        # print('height shape:',cube.coord('height').shape[0])

        # print(cube)
        # cube=self.fix_height2m(cube,cubes)

        # cube= self.fix_height_name(cube)
        

        return CubeList([cube_checked])


class psl(NativeDatasetFix):

    def fix_metadata(self, cubes):

        master_map_path='./master_map.csv'

        with open (master_map_path,'r') as map:
            reader=csv.reader(map, delimiter=',')
            for raw in reader:
                if raw[0]=='pr':
                    pr_map=raw
                    break

        # original_short_name='air_temperature'
        original_short_name='fld_s16i222'

        cube= self.get_cube(cubes, var_name=original_short_name)

        print('Successfully get the cube(psl)')

        # print('self.vardef:',self.vardef.dimensions)

        # print('height shape:',cube.coord('height').shape[0])

        # print(cube)
        # cube=self.fix_height2m(cube,cubes)

        # cube= self.fix_height_name(cube)
        

        return CubeList([cube])

class sftlf(NativeDatasetFix):

    def fix_metadata(self,cubes):
        original_short_name='fld_s03i395'

        cube= self.get_cube(cubes, var_name=original_short_name)

        print('Successfully get the cube(sftlf)')

        return CubeList([cube])


