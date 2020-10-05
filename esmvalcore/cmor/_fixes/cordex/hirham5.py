"""Fixes for DMI-HIRHAM5 model."""
import iris

from ..common import ClFixHybridPressureCoord
from ..fix import Fix
from ..shared import add_scalar_height_coord

import IPython
from traitlets.config import get_config
c = get_config()
c.InteractiveShellEmbed.colors = "Linux"

import numpy as np
from esmvalcore.preprocessor._shared import guess_bounds

coord_sys_rotated = iris.coord_systems.RotatedGeogCS(39.25, -162.0)
grid_lat_11 = iris.coords.DimCoord(np.linspace(-23.375, 21.835, 412),
                                   var_name='rlat',
                                   standard_name='grid_latitude',
                                   long_name='latitude in rotated-pole grid',
                                   units='degrees',
                                   coord_system=coord_sys_rotated)
grid_lon_11 = iris.coords.DimCoord(np.linspace(-28.375, 18.155, 424),
                                   var_name='rlon',
                                   standard_name='grid_longitude',
                                   long_name='longitude in rotated-pole grid',
                                   units='degrees',
                                   coord_system=coord_sys_rotated)
grid_lat_44 = iris.coords.DimCoord(np.linspace(-23.21, 21.67, 103),
                                   var_name='rlat',
                                   standard_name='grid_latitude',
                                   long_name='latitude in rotated-pole grid',
                                   units='degrees',
                                   coord_system=coord_sys_rotated)
grid_lon_44 = iris.coords.DimCoord(np.linspace(-28.21, 17.99, 106),
                                   var_name='rlon',
                                   standard_name='grid_longitude',
                                   long_name='longitude in rotated-pole grid',
                                   units='degrees',
                                   coord_system=coord_sys_rotated)


class Tas(Fix):
    """Fixes for tas."""

    def fix_metadata(self, cubes):
        """
        Add missing height2m coordinate.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList

        """
        for cube in cubes:
            add_scalar_height_coord(cube)

        return cubes


class AllVars(Fix):
    """Fixes for all variables."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        fixed_cubes = iris.cube.CubeList()

        # IPython.embed(config=c)

        cubes = [c for c in cubes if c.standard_name == self.vardef.standard_name]
        for cube in cubes:
            if cube.attributes['CORDEX_domain'] == 'EUR-11':
                if len(cube.dim_coords) < 2 and len(cube.shape) > 1:
                    cube.add_dim_coord(grid_lat_11,
                                       cube.shape.index(grid_lat_11.shape[0]))
                    cube.add_dim_coord(grid_lon_11,
                                       cube.shape.index(grid_lon_11.shape[0]))
                else:
                    if cube.coord('grid_latitude') != grid_lat_11:
                        cube.replace_coord(grid_lat_11)
                    if cube.coord('grid_longitude') != grid_lon_11:
                        cube.replace_coord(grid_lon_11)
                cube = guess_bounds(cube, ['grid_latitude', 'grid_longitude'])
                cube.data = cube.core_data().astype('float32')
            elif cube.attributes['CORDEX_domain'] == 'EUR-44':
                if len(cube.dim_coords) < 2 and len(cube.shape) > 1:
                    cube.add_dim_coord(grid_lat_44,
                                       cube.shape.index(grid_lat_44.shape[0]))
                    cube.add_dim_coord(grid_lon_44,
                                       cube.shape.index(grid_lon_44.shape[0]))
                else:
                    if cube.coord('grid_latitude') != grid_lat_44:
                        cube.replace_coord(grid_lat_44)
                    if cube.coord('grid_longitude') != grid_lon_44:
                        cube.replace_coord(grid_lon_44)
                cube = guess_bounds(cube, ['grid_latitude', 'grid_longitude'])
                # cube.data = cube.core_data().astype('float32')
            fixed_cubes.append(cube)

    # def fix_metadata(self, cubes):
    #     """Fix metadata."""
    #     fixed_cubes = iris.cube.CubeList()
    #     for cube in cubes:
    #         cube.data = cube.core_data().astype('float32')
    #         fixed_cubes.append(cube)

    #     return fixed_cubes


        # IPython.embed(config=c)
        #      # No idea, but there is only one cube
        # for cube in cubes:
        #     if len(cube.dim_coords) < 2 and len(cube.shape) > 1:
        #         print('here')
        #         IPython.embed(config=c)

        return fixed_cubes


    # def fix_file(file, short_name, project, dataset, mip, output_dir):
    #     """
    #     Fix files before ESMValTool can load them.

    #     This fixes are only for issues that prevent iris from loading the cube or
    #     that cannot be fixed after the cube is loaded.

    #     Original files are not overwritten.

    #     Parameters
    #     ----------
    #     file: str
    #         Path to the original file
    #     short_name: str
    #         Variable's short name
    #     project: str
    #     dataset:str
    #     output_dir: str
    #         Output directory for fixed files

    #     Returns
    #     -------
    #     str:
    #         Path to the fixed file

    #     """
    #     for fix in Fix.get_fixes(
    #             project=project, dataset=dataset, mip=mip, short_name=short_name):
    #         file = fix.fix_file(file, output_dir)
    #     return file
