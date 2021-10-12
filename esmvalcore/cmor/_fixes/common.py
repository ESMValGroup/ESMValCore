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
from esmvalcore.preprocessor._shared import guess_bounds


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


class RCMFixes(Fix):
    """Generic fixes for the RCMs."""

    def fix_metadata(self, cubes):
        """Fix metatdata.

        Calculate missing latitude/longitude boundaries using interpolation.

        Parameters
        ----------
        cubes: iris.cube.CubeList
            List of cubes to fix

        Returns
        -------
        iris.cube.CubeList
            Fixed cubes

        """
        cube = self.get_cube_from_list(cubes)

        # ensure correct var names for lat and lon
        lats = cube.coords("latitude")
        if lats:
            lat = cube.coord("latitude")
            lat.var_name = "lat"

        lons = cube.coords("longitude")
        if lons:
            lon = cube.coord("longitude")
            lon.var_name = "lon"

        coord_names = [coord.standard_name for coord in cube.coords()]
        x_name = cube.coord(axis='x', dim_coords=True).standard_name
        y_name = cube.coord(axis='y', dim_coords=True).standard_name
        if x_name != 'longitude' and y_name != 'latitude':
            # Fix some issues with Lambert conformal coordinates
            self.fix_coordinate_system(cube)

            # get the grid bounds
            cube = guess_bounds(cube, [x_name, y_name])

            # get the lat lon the for global coord system (if missing)
            if 'latitude' not in coord_names or \
               'longitude' not in coord_names:
                cube = self.get_lat_lon_coordinates(cube)
            # get the lat lon bounds for the global coord system (if missing)
            if not cube.coord('latitude').has_bounds() and \
               not cube.coord('longitude').has_bounds():
                cube = self.get_lat_lon_bounds(cube)

            cube = self.check_grid_differences(cube)

        return cubes

    @staticmethod
    def check_grid_differences(cube):
        """
        Derive lat and lon coordinates from grid coordinates.

        It also warns about the maximum differences
        """
        x_coord = cube.coord(axis='x', dim_coords=True)
        y_coord = cube.coord(axis='y', dim_coords=True)
        glon, glat = np.meshgrid(x_coord.points, y_coord.points)

        global_coord_sys = iris.coord_systems.GeogCS(
            iris.fileformats.pp.EARTH_RADIUS)
        grid_coord_sys = x_coord.coord_system

        if isinstance(grid_coord_sys, iris.coord_systems.RotatedGeogCS):
            gr_type = "rotated"
            glon, glat = np.meshgrid(x_coord.points, y_coord.points)
            lons, lats = iris.analysis.cartography.unrotate_pole(
                glon, glat,
                grid_coord_sys.grid_north_pole_longitude,
                grid_coord_sys.grid_north_pole_latitude
            )
        elif isinstance(grid_coord_sys, iris.coord_systems.LambertConformal):
            gr_type = "lcc"
            ccrs_global = global_coord_sys.as_cartopy_crs()
            xyz = ccrs_global.transform_points(grid_coord_sys.as_cartopy_crs(),
                                               glon, glat)
            lons = xyz[:, :, 0]
            lats = xyz[:, :, 1]
        else:
            emsg = 'Unknown coordinate system, got {!r}.'
            raise NotImplementedError(
                emsg.format(type(x_coord.grid_coord_sys)))

        dif = np.sqrt(np.square(lats - cube.coord('latitude').points) +
                      np.square(lons - cube.coord('longitude').points))
        if dif.max() != 0:
            logger.warning("There are diffs. betw. %s and lat-lon.", gr_type)
            logger.warning("Max diff: %s ; Min diff: %s", dif.max(), dif.min())
            cube.attributes.update({'grid_max_spacial_diff': dif.max()})
            cube.attributes.update({'grid_min_spacial_diff': dif.min()})
        return cube

    @staticmethod
    def get_lat_lon_bounds(cube):
        """
        Derive lat and lon bounds from grid coordinates.

        CMOR standard for 2-D coordinate bounds is indexing the four vertices
        as follows: 0=(j-1,i-1), 1=(j-1,i+1), 2=(j+1,i+1), 3=(j+1,i-1).
        """
        x_coord = cube.coord(axis='x', dim_coords=True)
        y_coord = cube.coord(axis='y', dim_coords=True)
        glon, glat = np.meshgrid(x_coord.bounds, y_coord.bounds)

        global_coord_sys = iris.coord_systems.GeogCS(
            iris.fileformats.pp.EARTH_RADIUS)
        grid_coord_sys = x_coord.coord_system

        if isinstance(grid_coord_sys, iris.coord_systems.RotatedGeogCS):
            lon_bnds, lat_bnds = iris.analysis.cartography.unrotate_pole(
                glon, glat,
                grid_coord_sys.grid_north_pole_longitude,
                grid_coord_sys.grid_north_pole_latitude
            )
        elif isinstance(grid_coord_sys, iris.coord_systems.LambertConformal):
            ccrs_global = global_coord_sys.as_cartopy_crs()
            xyz = ccrs_global.transform_points(
                grid_coord_sys.as_cartopy_crs(), glon, glat)
            lon_bnds = xyz[:, :, 0]
            lat_bnds = xyz[:, :, 1]
        else:
            emsg = 'Unknown coordinate system, got {!r}.'
            raise NotImplementedError(
                emsg.format(type(x_coord.grid_coord_sys)))

        if not cube.coord('latitude').has_bounds():
            lat_bnds = [lat_bnds[::2, ::2], lat_bnds[::2, 1::2],
                        lat_bnds[1::2, 1::2], lat_bnds[1::2, ::2]]
            lat_bnds = np.array(lat_bnds)
            lat_bnds = np.moveaxis(lat_bnds, 0, -1)
            cube.coord('latitude').bounds = lat_bnds

        if not cube.coord('longitude').has_bounds():
            lon_bnds = [lon_bnds[::2, ::2], lon_bnds[::2, 1::2],
                        lon_bnds[1::2, 1::2], lon_bnds[1::2, ::2]]
            lon_bnds = np.array(lon_bnds)
            lon_bnds = np.moveaxis(lon_bnds, 0, -1)
            cube.coord('longitude').bounds = lon_bnds

        return cube

    @staticmethod
    def get_lat_lon_coordinates(cube):
        """Derive lat and lon coordinates from grid coordinates."""
        coord_names = [coord.standard_name for coord in cube.coords()]
        x_coord = cube.coord(axis='x', dim_coords=True)
        y_coord = cube.coord(axis='y', dim_coords=True)
        glon, glat = np.meshgrid(x_coord.points, y_coord.points)

        global_coord_sys = iris.coord_systems.GeogCS(
            iris.fileformats.pp.EARTH_RADIUS)
        grid_coord_sys = x_coord.coord_system

        if isinstance(grid_coord_sys, iris.coord_systems.RotatedGeogCS):
            glon, glat = np.meshgrid(x_coord.points, y_coord.points)
            lons, lats = iris.analysis.cartography.unrotate_pole(
                glon, glat,
                grid_coord_sys.grid_north_pole_longitude,
                grid_coord_sys.grid_north_pole_latitude
            )
        elif isinstance(grid_coord_sys, iris.coord_systems.LambertConformal):
            ccrs_global = global_coord_sys.as_cartopy_crs()
            xyz = ccrs_global.transform_points(
                grid_coord_sys.as_cartopy_crs(), glon, glat)
            lons = xyz[:, :, 0]
            lats = xyz[:, :, 1]
        else:
            emsg = 'Unknown coordinate system, got {!r}.'
            raise NotImplementedError(
                emsg.format(type(x_coord.grid_coord_sys)))

        if 'latitude' not in coord_names:
            lat = iris.coords.AuxCoord(lats,
                                       var_name='lat',
                                       standard_name='latitude',
                                       units='degrees',
                                       coord_system=global_coord_sys)
            RCMFixes._add_coordinate(lat, cube)

        if 'longitude' not in coord_names:
            lon = iris.coords.AuxCoord(lons,
                                       var_name='lon',
                                       standard_name='longitude',
                                       units='degrees',
                                       coord_system=global_coord_sys)
            RCMFixes._add_coordinate(lon, cube)

        return cube

    @staticmethod
    def _add_coordinate(lat, cube):
        if lat.shape[0] == lat.shape[1]:
            data_dims = np.where(np.array(cube.shape) == lat.shape[0])[0]
        else:
            data_dims = [cube.shape.index(lat.shape[0]),
                         cube.shape.index(lat.shape[1])]

        cube.add_aux_coord(lat, data_dims=data_dims)

    @staticmethod
    def fix_coordinate_system(cube):
        """Fix LambertConformal."""
        x_coord = cube.coord(axis='x', dim_coords=True)
        y_coord = cube.coord(axis='y', dim_coords=True)

        global_coord_sys = iris.coord_systems.GeogCS(
            iris.fileformats.pp.EARTH_RADIUS)
        grid_coord_sys = x_coord.coord_system

        if isinstance(grid_coord_sys, iris.coord_systems.LambertConformal):
            # there are some badly defined coordinate systems:
            if grid_coord_sys.ellipsoid:
                if grid_coord_sys.ellipsoid.semi_minor_axis == float('-inf'):
                    grid_coord_sys.ellipsoid.semi_minor_axis = None
            else:
                grid_coord_sys.ellipsoid = global_coord_sys

            # badly defined units
            if x_coord.units == 'km':
                x_coord.convert_units('meter')
            if y_coord.units == 'km':
                y_coord.convert_units('meter')

            # and badly defined offsets
            if x_coord[0].points[0] == 0:
                x_offset = x_coord.points.mean()
                grid_coord_sys.false_easting = x_offset
            if y_coord[0].points[0] == 0:
                y_offset = y_coord.points.mean()
                grid_coord_sys.false_northing = y_offset

            # This is for
            # - {dataset: ICTP-RegCM4-3, driver: MOHC-HadGEM2-ES}
            # LambertConformal of the netcdf gives a false_easting though
            # x_offset is 0, setting false_easting to zero gives much smaller
            # differences to latitude longitude in the netcdf
            if x_coord.points.mean() == 0 and \
               grid_coord_sys.false_easting is not None:
                grid_coord_sys.false_easting = 0
            if y_coord.points.mean() == 0 and \
               grid_coord_sys.false_northing is not None:
                grid_coord_sys.false_northing = 0

        return cube
