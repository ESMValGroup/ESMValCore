"""Fixes for CNRM ALADIN53 model."""
import copy

import iris
import numpy as np

from esmvalcore.preprocessor._shared import guess_bounds

from ..fix import Fix

class Orog(Fix):
    """Fixes for tas."""

    def fix_metadata(self, cubes):
        """
        Issue: For some reason, orog coordinates deviate by a factor of 4.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes

        Returns
        -------
        iris.cube.CubeList

        """
        fixed_cubes = iris.cube.CubeList()
        for cube in cubes:
            for cname, cindex in zip(['projection_x_coordinate',
                                      'projection_y_coordinate'],
                                     [1, 0]):
                coord = copy.deepcopy(cube.coords(dimensions=cindex)[0])
                coord = coord / 4.
                cube.remove_coord(cname)
                cube.add_dim_coord(coord, cindex)
            fixed_cubes.append(cube)

        return fixed_cubes


class AllVars(Fix):
    """Fixes for all variables."""

    def fix_metadata(self, cubes):
        """Fix metadata.

        Fails to create 'x' and 'y' dimension coordinate:
        DimCoord points array must be strictly monotonic"""
        fixed_cubes = iris.cube.CubeList()

        for cube in cubes:
            if cube.attributes['CORDEX_domain'] == 'EUR-11':
                if len(cube.dim_coords) < 2 and len(cube.shape) > 1:
                    for cname, cindex in zip(['projection_x_coordinate',
                                              'projection_y_coordinate'],
                                             [2, 1]):
                        coord = cube.coord(cname)
                        coord.points = coord.points.astype('float32')
                        coord.points = np.linspace(coord.points[0],
                                                   coord.points[-1],
                                                   coord.shape[0])
                        newcoord = iris.coords.DimCoord(coord.points.data,
                                            var_name=coord.var_name,
                                            standard_name=coord.standard_name,
                                            long_name=coord.long_name,
                                            units=coord.units,
                                            coord_system=coord.coord_system)
                        cube.remove_coord(cname)
                        cube.add_dim_coord(newcoord, cindex)
                cube = guess_bounds(cube, ['projection_x_coordinate',
                                           'projection_y_coordinate'])
            fixed_cubes.append(cube)
        return fixed_cubes
