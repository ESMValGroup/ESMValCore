"""Common fixes used for multiple datasets."""
import logging

import iris
import numpy as np
from scipy.ndimage import map_coordinates

from .fix import Fix
from .shared import (
    add_plev_from_altitude,
    add_scalar_typesi_coord,
    fix_bounds,
)

logger = logging.getLogger(__name__)


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

    def fix_metadata(self, cubes):
        """Fix ``latitude`` and ``longitude`` (metadata and bounds).

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList

        """
        cube = self.get_cube_from_list(cubes)
        if cube.ndim != 3:
            logger.warning(
                "OceanFixGrid is designed to work on any data with an "
                "irregular ocean grid, but it was only tested on 3D (time, "
                "latitude, longitude) data so far; got %dD data", cube.ndim)

        # Get dimensional coordinates. Note:
        # - First dimension i -> X-direction (= longitude)
        # - Second dimension j -> Y-direction (= latitude)
        (j_dim, i_dim) = sorted(set(
            cube.coord_dims(cube.coord('latitude', dim_coords=False)) +
            cube.coord_dims(cube.coord('longitude', dim_coords=False))
        ))
        i_coord = cube.coord(dim_coords=True, dimensions=i_dim)
        j_coord = cube.coord(dim_coords=True, dimensions=j_dim)

        # Fix metadata of coordinate i
        i_coord.var_name = 'i'
        i_coord.standard_name = None
        i_coord.long_name = 'cell index along first dimension'
        i_coord.units = '1'
        i_coord.circular = False

        # Fix metadata of coordinate j
        j_coord.var_name = 'j'
        j_coord.standard_name = None
        j_coord.long_name = 'cell index along second dimension'
        j_coord.units = '1'

        # Fix points and bounds of index coordinates i and j
        for idx_coord in (i_coord, j_coord):
            idx_coord.points = np.arange(len(idx_coord.points))
            idx_coord.bounds = None
            idx_coord.guess_bounds()

        # Calculate latitude/longitude vertices by interpolation.
        # Following the CF conventions (see
        # cfconventions.org/cf-conventions/cf-conventions.html#cell-boundaries)
        # we go counter-clockwise around the cells and construct a grid of
        # index values which are in turn used to interpolate longitudes and
        # latitudes in the midpoints between the cell centers.
        lat_vertices = []
        lon_vertices = []
        for (j, i) in [(0, 0), (0, 1), (1, 1), (1, 0)]:
            (j_v, i_v) = np.meshgrid(j_coord.bounds[:, j],
                                     i_coord.bounds[:, i],
                                     indexing='ij')
            lat_vertices.append(
                map_coordinates(cube.coord('latitude').points,
                                [j_v, i_v],
                                mode='nearest'))
            lon_vertices.append(
                map_coordinates(cube.coord('longitude').points,
                                [j_v, i_v],
                                mode='wrap'))
        lat_vertices = np.array(lat_vertices)
        lon_vertices = np.array(lon_vertices)
        lat_vertices = np.moveaxis(lat_vertices, 0, -1)
        lon_vertices = np.moveaxis(lon_vertices, 0, -1)

        # Copy vertices to cube
        cube.coord('latitude').bounds = lat_vertices
        cube.coord('longitude').bounds = lon_vertices

        return iris.cube.CubeList([cube])


class SiconcFixScalarCoord(Fix):
    """Fixes for siconc."""

    def fix_metadata(self, cubes):
        """Add typesi coordinate.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList

        """
        cube = self.get_cube_from_list(cubes)
        add_scalar_typesi_coord(cube)
        return iris.cube.CubeList([cube])
