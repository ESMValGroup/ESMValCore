"""Common fixes used for multiple datasets."""
import iris
import numpy as np
from scipy.ndimage import map_coordinates

from .fix import Fix
from .shared import add_plev_from_altitude, fix_bounds


class ClFixHybridHeightCoord(Fix):
    """Fixes for ``cl`` regarding hybrid sigma height coordinates."""

    def fix_metadata(self, cubes):
        """Fix hybrid sigma height coordinate and add pressure levels.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes which need to be fixed.

        Returns
        -------
        iris.cube.CubeList

        """
        cube = self.get_cube_from_list(cubes)

        # Remove all existing aux_factories
        for aux_factory in cube.aux_factories:
            cube.remove_aux_factory(aux_factory)

        # Fix bounds
        fix_bounds(cube, cubes, ('lev', 'b'))

        # Add aux_factory again
        height_coord_factory = iris.aux_factory.HybridHeightFactory(
            delta=cube.coord(var_name='lev'),
            sigma=cube.coord(var_name='b'),
            orography=cube.coord(var_name='orog'),
        )
        cube.add_aux_factory(height_coord_factory)

        # Add pressure level coordinate
        add_plev_from_altitude(cube)

        return iris.cube.CubeList([cube])


class ClFixHybridPressureCoord(Fix):
    """Fixes for ``cl`` regarding hybrid sigma pressure coordinates."""

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
        cube = self.get_cube_from_list(cubes)

        # Remove all existing aux_factories
        for aux_factory in cube.aux_factories:
            cube.remove_aux_factory(aux_factory)

        # Fix bounds
        coords_to_fix = ['b']
        try:
            cube.coord(var_name='a')
            coords_to_fix.append('a')
        except iris.exceptions.CoordinateNotFoundError:
            coords_to_fix.append('ap')
        fix_bounds(cube, cubes, coords_to_fix)

        # Fix bounds for ap if only a is given in original file
        # This was originally done by iris, but it has to be repeated since
        # a has bounds now
        ap_coord = cube.coord(var_name='ap')
        if ap_coord.bounds is None:
            cube.remove_coord(ap_coord)
            a_coord = cube.coord(var_name='a')
            p0_coord = cube.coord(var_name='p0')
            ap_coord = a_coord * p0_coord.points[0]
            ap_coord.units = a_coord.units * p0_coord.units
            ap_coord.rename('vertical pressure')
            ap_coord.var_name = 'ap'
            cube.add_aux_coord(ap_coord, cube.coord_dims(a_coord))

        # Add aux_factory again
        pressure_coord_factory = iris.aux_factory.HybridPressureFactory(
            delta=ap_coord,
            sigma=cube.coord(var_name='b'),
            surface_air_pressure=cube.coord(var_name='ps'),
        )
        cube.add_aux_factory(pressure_coord_factory)

        # Remove attributes from Surface Air Pressure coordinate
        cube.coord(var_name='ps').attributes = {}

        return iris.cube.CubeList([cube])


class OceanFixGrid(Fix):
    """Fixes for tos, siconc in FGOALS-g3."""

    def fix_data(self, cube):
        """
        Fix data.

        Calculate missing latitude/longitude boundaries using interpolation.
        Based on a similar fix for BCC-CSM2-MR.

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

        # Guess coordinate bounds in rlat, rlon (following BCC-CSM2-MR-1).
        rlat_idx_bnds = np.zeros((len(rlat), 2))
        rlat_idx_bnds[:, 0] = np.arange(len(rlat)) - 0.5
        rlat_idx_bnds[:, 1] = np.arange(len(rlat)) + 0.5
        rlat_idx_bnds[0, 0] = 0.
        rlat_idx_bnds[len(rlat) - 1, 1] = len(rlat)
        rlon_idx_bnds = np.zeros((len(rlon), 2))
        rlon_idx_bnds[:, 0] = np.arange(len(rlon)) - 0.5
        rlon_idx_bnds[:, 1] = np.arange(len(rlon)) + 0.5

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

    def fix_metadata(self, cubes):
        """
        Rename ``var_name`` of 1D-``latitude`` and 1D-``longitude``.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList
        """
        cube = self.get_cube_from_list(cubes)
        lat_coord = cube.coord('cell index along second dimension',
                               dimensions=(1, ))
        lon_coord = cube.coord('cell index along first dimension',
                               dimensions=(2, ))
        lat_coord.standard_name = None
        lat_coord.long_name = 'grid_latitude'
        lat_coord.var_name = 'i'
        lat_coord.units = '1'
        lon_coord.standard_name = None
        lon_coord.long_name = 'grid_longitude'
        lon_coord.var_name = 'j'
        lon_coord.units = '1'
        lon_coord.circular = False
        # FGOALS-g3 data contain latitude and longitude data set to
        # >1e30 in some places. Set to 0. to avoid problem in check.py.
        cube.coord('latitude').points[cube.coord('latitude').points > 1000.]\
            = 0.
        cube.coord('longitude').points[cube.coord('longitude').points > 1000.]\
            = 0.
        return cubes
