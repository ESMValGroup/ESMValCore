"""Fixes for DMI-HIRHAM5 model."""
import iris
import numpy as np

from ..fix import Fix
from ..shared import add_scalar_height_coord

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
            fixed_cubes.append(cube)
        return fixed_cubes
