import logging

import csv

from iris.cube import CubeList

from ..native_datasets import NativeDatasetFix

from esmvalcore.cmor.check import cmor_check

import os

logger = logging.getLogger(__name__)

class tas(NativeDatasetFix):
    '''
    Fix variable(tas) only
    '''

    def __init__(self, vardef, extra_facets, session, frequency):
        '''
        Initialise some class variable
        Heritage from native_dataset
        '''

        super().__init__(vardef,extra_facets,session,frequency)

        self.cube=None

        self.current_dir=os.path.dirname(__file__)
        
    def _fix_height_name(self):
        '''
        Fix variable name of coordinate 'height' 
        '''
        if self.cube.coord('height').var_name!='height':
            self.cube.coord('height').var_name='height'
    
    def _fix_long_name(self):
        '''
        Fix variable long_name
        '''
        self.cube.long_name ='Near-Surface Air Temperature'

    def _fix_var_name(self):
        '''
        Fix variable long_name
        '''
        self.cube.var_name='tas'
    
    def fix_coord_system(self):
        '''
        delete coord_system to make it cna be merged with other cmip dataset by iris.CubeList.merge_dube
        '''
        for dim in self.cube.dim_coords:
            if dim.coord_system!=None:
                self.cube.coord(dim.standard_name).coord_system=None
    
    def _load_master_map(self,short_name):
        '''
        Master map is a supplimentary file for how to convert access variable to cmip data
        
        Parameters
        ----------
        short_name : str
            short name of variable.

        Returns
        -------
        list which contain supplimentary imformation of the variable

        '''
        master_map_path=f'{self.current_dir}/master_map.csv'
        with open (master_map_path,'r') as map:
            reader=csv.reader(map, delimiter=',')
            for raw in reader:
                if raw[0]==short_name:
                    return raw
    
    def fix_metadata(self, cubes):
        """
        Fix name of coordinate(height), long name and variable name of variable(tas).

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList

        """

        row=self._load_master_map(self.vardef.short_name)

        original_short_name=row[0]

        self.cube= self.get_cube(cubes, var_name=original_short_name)

        self.fix_height_name()

        self.fix_long_name()

        self.fix_var_name()

        self.fix_coord_system()

        cube_checked= cmor_check(cube=self.cube,cmor_table='CMIP6',mip='Amon',short_name='tas',check_level=1)

        return CubeList([cube_checked])


class pr(NativeDatasetFix):
    '''
    Fix variable(pr) only
    '''

    def __init__(self,vardef,
        extra_facets,
        session,
        frequency):
        '''
        Initialise some class variable
        Heritage from native_dataset
        '''

        super().__init__(vardef,extra_facets,session,frequency)

        self.cube=None

        self.current_dir=os.path.dirname(__file__)
    

    def fix_var_name(self):
        '''
        Fix variable long_name
        '''
        self.cube.var_name='pr'
    
    def fix_long_name(self):
        '''
        Fix variable long_name
        '''
        self.cube.long_name ='Precipitation'


    def fix_coord_system(self):
        '''
        delete coord_system to make it cna be merged with other cmip dataset by iris.CubeList.merge_dube
        '''
        for dim in self.cube.dim_coords:
            if dim.coord_system!=None:
                self.cube.coord(dim.standard_name).coord_system=None
    
    def _load_master_map(self,short_name):
        '''
        Master map is a supplimentary file for how to convert access variable to cmip data
        
        Parameters
        ----------
        short_name : str
            short name of variable.

        Returns
        -------
        list which contain supplimentary imformation of the variable

        '''
        master_map_path=f'{self.current_dir}/master_map.csv'
        with open (master_map_path,'r') as map:
            reader=csv.reader(map, delimiter=',')
            for raw in reader:
                if raw[0]==short_name:
                    return raw

    def fix_metadata(self, cubes):
        """
        Fix name of coordinate(height), long name and variable name of variable(tas).

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList

        """

        row=self._load_master_map(self.vardef.short_name)

        original_short_name=row[1]

        cube= self.get_cube(cubes, var_name=original_short_name)

        self.fix_var_name()

        self.fix_long_name()

        self.fix_coord_system()

        cube_checked= cmor_check(cube=cube,cmor_table='CMIP6',mip='Amon',short_name='pr',check_level=1)

        return CubeList([cube_checked])

