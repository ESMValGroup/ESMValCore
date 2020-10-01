"""Fixes for CORDEX project"""
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.ndimage import map_coordinates

from esmvalcore.cmor.fix import Fix


class AllVars(Fix):
    """Generic fixes for the CORDEX project"""

    def fix_metadata(self, cubes):
        """Fix metatdata.

        Calculate missing latitude/longitude boundaries using interpolation.

        Parameters
        ----------
        cubes: iris.cube.CubeList

        Returns
        -------
        iris.cube.CubeList

        """
        for cube in cubes:
            if not cube.coords('latitude'):
                continue
            if not cube.coords('longitude'):
                continue
            if cube.coord('latitude').has_bounds() and \
                    cube.coord('longitude').has_bounds():
                continue

            rlat_idx_bnds = self._get_points_and_bounds(
                cube.coord('grid_latitude'))
            rlon_idx_bnds = self._get_points_and_bounds(
                cube.coord('grid_longitude'))

            # Calculate latitude/longitude vertices by interpolation
            lat_vertices = []
            lon_vertices = []
            for (i, j) in [(0, 0), (0, 1), (1, 1), (1, 0)]:
                (rlat_v, rlon_v) = np.meshgrid(
                    rlat_idx_bnds[:, i], rlon_idx_bnds[:, j], indexing='ij')
                lat_vertices.append(
                    map_coordinates(
                        cube.coord('latitude').points, [rlat_v, rlon_v],
                        mode='nearest'))
                lon_vertices.append(
                    map_coordinates(
                        cube.coord('longitude').points, [rlat_v, rlon_v],
                        mode='wrap'))
            lat_vertices = np.array(lat_vertices)
            lon_vertices = np.array(lon_vertices)
            lat_vertices = np.moveaxis(lat_vertices, 0, -1)
            lon_vertices = np.moveaxis(lon_vertices, 0, -1)

            # Copy vertices to cube
            cube.coord('latitude').bounds = lat_vertices
            cube.coord('longitude').bounds = lon_vertices

        return cubes

    def _get_points_and_bounds(self, coord):
        coord = coord.copy()
        coord.guess_bounds()
        points_to_idx = InterpolatedUnivariateSpline(
            coord.points, np.arange(len(coord.points)), k=1)
        bounds = points_to_idx(coord.bounds)
        return bounds
