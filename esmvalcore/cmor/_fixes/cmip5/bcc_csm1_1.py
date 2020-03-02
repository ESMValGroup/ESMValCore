"""Fixes for bcc-csm1-1."""
import iris
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.ndimage import map_coordinates

from ..fix import Fix
from ..shared import fix_bounds


class Cl(Fix):
    """Fixes for cl."""

    def fix_metadata(self, cubes):
        """Fix hybrid sigma pressure coordinate.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes which need to be fixed.

        Returns
        -------
        iris.cube.CubeList

        """
        cl_cube = self.get_cube_from_list(cubes)

        # Remove all existing aux_factories
        for aux_factory in cl_cube.aux_factories:
            cl_cube.remove_aux_factory(aux_factory)

        # Fix bounds
        coords_to_fix = ['b']
        try:
            cl_cube.coord(var_name='a')
            coords_to_fix.append('a')
        except iris.exceptions.CoordinateNotFoundError:
            coords_to_fix.append('ap')
        fix_bounds(cl_cube, cubes, coords_to_fix)

        # Fix bounds for ap if only a is given in original file
        ap_coord = cl_cube.coord(var_name='ap')
        if ap_coord.bounds is None:
            cl_cube.remove_coord(ap_coord)
            a_coord = cl_cube.coord(var_name='a')
            p0_coord = cl_cube.coord(var_name='p0')
            ap_coord = a_coord * p0_coord.points[0]
            ap_coord.units = a_coord.units * p0_coord.units
            ap_coord.rename('vertical pressure')
            ap_coord.var_name = 'ap'
            cl_cube.add_aux_coord(ap_coord, cl_cube.coord_dims(a_coord))

        # Add aux_factory again
        pressure_coord_factory = iris.aux_factory.HybridPressureFactory(
            delta=ap_coord,
            sigma=cl_cube.coord(var_name='b'),
            surface_air_pressure=cl_cube.coord(var_name='ps'),
        )
        cl_cube.add_aux_factory(pressure_coord_factory)

        return iris.cube.CubeList([cl_cube])


class Tos(Fix):
    """Fixes for tos."""

    def fix_data(self, cube):
        """Fix data.

        Calculate missing latitude/longitude boundaries using interpolation.

        Parameters
        ----------
        cube: iris.cube.Cube
            Input cube to fix.

        Returns
        -------
        iris.cube.Cube

        """
        rlat = cube.coord('grid_latitude').points
        rlon = cube.coord('grid_longitude').points

        # Transform grid latitude/longitude to array indices [0, 1, 2, ...]
        rlat_to_idx = InterpolatedUnivariateSpline(rlat,
                                                   np.arange(len(rlat)),
                                                   k=1)
        rlon_to_idx = InterpolatedUnivariateSpline(rlon,
                                                   np.arange(len(rlon)),
                                                   k=1)
        rlat_idx_bnds = rlat_to_idx(cube.coord('grid_latitude').bounds)
        rlon_idx_bnds = rlon_to_idx(cube.coord('grid_longitude').bounds)

        # Calculate latitude/longitude vertices by interpolation
        lat_vertices = []
        lon_vertices = []
        for (i, j) in [(0, 0), (0, 1), (1, 1), (1, 0)]:
            (rlat_v, rlon_v) = np.meshgrid(rlat_idx_bnds[:, i],
                                           rlon_idx_bnds[:, j],
                                           indexing='ij')
            lat_vertices.append(
                map_coordinates(cube.coord('latitude').points,
                                [rlat_v, rlon_v],
                                mode='nearest'))
            lon_vertices.append(
                map_coordinates(cube.coord('longitude').points,
                                [rlat_v, rlon_v],
                                mode='wrap'))
        lat_vertices = np.array(lat_vertices)
        lon_vertices = np.array(lon_vertices)
        lat_vertices = np.moveaxis(lat_vertices, 0, -1)
        lon_vertices = np.moveaxis(lon_vertices, 0, -1)

        # Copy vertices to cube
        cube.coord('latitude').bounds = lat_vertices
        cube.coord('longitude').bounds = lon_vertices

        return cube
