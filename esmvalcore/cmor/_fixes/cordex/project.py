"""Fixes for CORDEX project"""
import logging
import numpy as np
import iris

from esmvalcore.cmor.fix import Fix
from esmvalcore.preprocessor._shared import guess_bounds

logger = logging.getLogger(__name__)


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
        cube = self.get_cube_from_list(cubes)

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


    def check_grid_differences(self, cube):
        """Derive lat and lon coordinates from grid coordinates.
        And warn about the maximum differences"""
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
                                    grid_coord_sys.grid_north_pole_latitude)
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

        dif = np.sqrt(np.square(lats - cube.coord('latitude').points) + \
                      np.square(lons - cube.coord('longitude').points))
        if dif.max() != 0:
            logger.warning(f"There are diffs. betw. {gr_type} and lat-lon.")
            logger.warning(f"Max diff: {dif.max()} ; Min diff: {dif.min()}")
            cube.attributes.update({'grid_max_spacial_diff': dif.max()})
            cube.attributes.update({'grid_min_spacial_diff': dif.min()})
        return cube


    def get_lat_lon_bounds(self, cube):
        """
        Derive lat and lon bounds from grid coordinates.
        CMOR standard for 2-D coordinate counds is indexing the four vertices
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
                                    grid_coord_sys.grid_north_pole_latitude)
        elif isinstance(grid_coord_sys, iris.coord_systems.LambertConformal):
            ccrs_global = global_coord_sys.as_cartopy_crs()
            xyz = ccrs_global.transform_points(grid_coord_sys.as_cartopy_crs(),
                                               glon, glat)
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


    def get_lat_lon_coordinates(self, cube):
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
                                    grid_coord_sys.grid_north_pole_latitude)
        elif isinstance(grid_coord_sys, iris.coord_systems.LambertConformal):
            ccrs_global = global_coord_sys.as_cartopy_crs()
            xyz = ccrs_global.transform_points(grid_coord_sys.as_cartopy_crs(),
                                               glon, glat)
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
            if lat.shape[0] == lat.shape[1]:
                data_dims = np.where(np.array(cube.shape) == lat.shape[0])[0]
            else:
                data_dims = [cube.shape.index(lat.shape[0]),
                             cube.shape.index(lat.shape[1])]

            cube.add_aux_coord(lat, data_dims=data_dims)

        if 'longitude' not in coord_names:
            lon = iris.coords.AuxCoord(lons,
                                       var_name='lon',
                                       standard_name='longitude',
                                       units='degrees',
                                       coord_system=global_coord_sys)
            if lon.shape[0] == lon.shape[1]:
                data_dims = np.where(np.array(cube.shape) == lon.shape[0])[0]
            else:
                data_dims = [cube.shape.index(lon.shape[0]),
                             cube.shape.index(lon.shape[1])]

            cube.add_aux_coord(lon, data_dims=data_dims)

        return cube


    def fix_coordinate_system(self, cube):
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
                grid_coord_sys.false_easting != None:
                grid_coord_sys.false_easting = 0
            if y_coord.points.mean() == 0 and \
                grid_coord_sys.false_northing != None:
                grid_coord_sys.false_northing = 0

        return cube
