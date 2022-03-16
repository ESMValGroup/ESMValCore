"""On-the-fly CMORizer for EMAC."""

import logging
import warnings
from datetime import datetime
from pathlib import Path
from shutil import copyfileobj
from urllib.parse import urlparse

import cf_units
import dask.array as da
import iris
import iris.coords
import iris.cube
import iris.util
import numpy as np
import requests

from esmvalcore.iris_helpers import (
    # add_leading_dim_to_cube,
    date2num,
)

from ..fix import Fix
from ..shared import add_scalar_height_coord, add_scalar_typesi_coord

logger = logging.getLogger(__name__)


CACHE_DIR = Path.home() / '.esmvaltool' / 'cache'
CACHE_VALIDITY = 7 * 24 * 60 * 60  # [s]; = 1 week
TIMEOUT = 5 * 60  # [s]; = 5 min
GRID_FILE_ATTR = 'grid_file_uri'


class EmacFix(Fix):
    """Base class for all EMAC fixes."""

    def get_cube(self, cubes, var_name=None):
        """Extract single cube."""
        if var_name is None:
            var_name = self.extra_facets.get('raw_name',
                                             self.vardef.short_name)
            if not cubes.extract(iris.NameConstraint(var_name=var_name)):
                raise ValueError(
                    f"Variable '{var_name}' used to extract "
                    f"'{self.vardef.short_name}' is not available in input "
                    f"file")
        return cubes.extract_cube(iris.NameConstraint(var_name=var_name))


class AllVars(EmacFix):
    """Fixes for all variables."""

    # def __init__(self, *args, **kwargs):
    #     """Initialize fix."""
    #     super().__init__(*args, **kwargs)
    #     self._horizontal_grids = {}


    def fix_data(self, cube):
        """Fix data."""
        # Fix mask by masking all values where the absolute value is greater
        # than a given threshold (affects mostly 3D variable)
        mask_threshold = 1e20
        cube.data = da.ma.masked_outside(
            cube.core_data(), -mask_threshold, mask_threshold,
        )
        return cube


    def fix_metadata(self, cubes):
        """Fix metadata."""
        cube = self.get_cube(cubes)

        # Fix time
        if 'time' in self.vardef.dimensions:
            self._fix_time(cube, cubes)

        # Fix pressure levels (considers plev19, plev39, etc.)
        for dim_name in self.vardef.dimensions:
            if 'plev' in dim_name:
                self._fix_plev(cube)
                break

        # here since the name of the z-coord varies from variable to variable
        # if cube.coords('height'):
        #     # In case a scalar height is required, remove it here (it will be
        #     # added later). This step here is designed to fix non-scalar height
        #     # coordinates.
        #     # Note: iris.util.squeeze is not used here since it might
        #     # accidentally squeeze other dimensions
        #     if (cube.coord('height').shape[0] == 1 and (
        #             'height2m' in self.vardef.dimensions or
        #             'height10m' in self.vardef.dimensions)):
        #         slices = [slice(None)] * cube.ndim
        #         slices[cube.coord_dims('height')[0]] = 0
        #         cube = cube[tuple(slices)]
        #         cube.remove_coord('height')
        #     else:
        #         cube = self._fix_height(cube, cubes)

        # Fix latitude
        # lat_idx = None
        if 'latitude' in self.vardef.dimensions:
            lat_name = self.extra_facets.get('latitude', 'latitude')
            # if not cube.coords(lat_name):
            #     self._add_coord_from_grid_file(cube, 'grid_latitude', lat_name)
            self._fix_lat(cube, lat_name)
            # lat_idx = cube.coord_dims('latitude')

        # Fix longitude
        # lon_idx = None
        if 'longitude' in self.vardef.dimensions:
            lon_name = self.extra_facets.get('longitude', 'longitude')
            # if not cube.coords(lon_name):
            #     self._add_coord_from_grid_file(cube, 'grid_longitude',
            #                                    lon_name)
            self._fix_lon(cube, lon_name)
            # lon_idx = cube.coord_dims('longitude')

        # Fix cell index for unstructured grid if necessary
        # fix_cell_index = all([
        #     lat_idx is not None,
        #     lon_idx is not None,
        #     lat_idx == lon_idx,
        #     len(lat_idx) == 1,
        # ])
        # if fix_cell_index:
        #     self._fix_unstructured_cell_index(cube, lat_idx)

        # Fix scalar coordinates
        self._fix_scalar_coords(cube)

        # Fix metadata of variable
        self._fix_var_metadata(cube)

        return iris.cube.CubeList([cube])

    # def get_horizontal_grid(self, grid_url):
    #     """Get horizontal grid."""
    #     # If already loaded, return the horizontal grid (cube)
    #     grid_name = str(grid_url)
    #     if grid_name in self._horizontal_grids:
    #         return self._horizontal_grids[grid_name]

    #     # Check if grid file has recently been downloaded and load it if
    #     # possible
    #     parsed_url = urlparse(grid_url)
    #     grid_path = CACHE_DIR / Path(parsed_url.path).name
    #     if grid_path.exists():
    #         mtime = grid_path.stat().st_mtime
    #         now = datetime.now().timestamp()
    #         age = now - mtime
    #         if age < CACHE_VALIDITY:
    #             logger.debug("Using cached ICON grid file '%s'", grid_path)
    #             self._horizontal_grids[grid_name] = self._load_cubes(grid_path)
    #             return self._horizontal_grids[grid_name]
    #         logger.debug("Existing cached ICON grid file '%s' is outdated",
    #                      grid_path)

    #     # Download file if necessary
    #     logger.debug("Attempting to download ICON grid file from '%s' to '%s'",
    #                  grid_url, grid_path)
    #     with requests.get(grid_url, stream=True, timeout=TIMEOUT) as response:
    #         response.raise_for_status()
    #         with open(grid_path, 'wb') as file:
    #             copyfileobj(response.raw, file)
    #     logger.info("Successfully downloaded ICON grid file from '%s' to '%s'",
    #                 grid_url, grid_path)

    #     self._horizontal_grids[grid_name] = self._load_cubes(grid_path)
    #     return self._horizontal_grids[grid_name]

    # def _add_coord_from_grid_file(self, cube, coord_name, target_coord_name):
    #     """Add latitude or longitude coordinate from grid file to cube."""
    #     allowed_coord_names = ('grid_latitude', 'grid_longitude')
    #     if coord_name not in allowed_coord_names:
    #         raise ValueError(
    #             f"coord_name must be one of {allowed_coord_names}, got "
    #             f"'{coord_name}'")
    #     if GRID_FILE_ATTR not in cube.attributes:
    #         raise ValueError(
    #             f"Cube does not contain coordinate '{coord_name}' nor the "
    #             f"attribute '{GRID_FILE_ATTR}' necessary to download the ICON "
    #             f"horizontal grid file:\n{cube}")
    #     grid_file_url = cube.attributes[GRID_FILE_ATTR]
    #     horizontal_grid = self.get_horizontal_grid(grid_file_url)

    #     # Use 'cell_area' as dummy cube to extract coordinates
    #     # Note: it might be necessary to expand this when more coord_names are
    #     # supported
    #     grid_cube = horizontal_grid.extract_cube(
    #         iris.NameConstraint(var_name='cell_area'))
    #     coord = grid_cube.coord(coord_name)

    #     # Find index of horizontal coordinate (try 'ncells' and unnamed
    #     # dimension)
    #     if cube.coords('ncells'):
    #         coord_dims = cube.coord_dims('ncells')
    #     else:
    #         n_unnamed_dimensions = cube.ndim - len(cube.dim_coords)
    #         if n_unnamed_dimensions != 1:
    #             raise ValueError(
    #                 f"Cannot determine coordinate dimension for coordinate "
    #                 f"'{coord_name}', cube does not contain coordinate "
    #                 f"'ncells' nor a single unnamed dimension:\n{cube}")
    #         coord_dims = ()
    #         for idx in range(cube.ndim):
    #             if not cube.coords(dimensions=idx, dim_coords=True):
    #                 coord_dims = (idx,)
    #                 break

    #     coord.standard_name = target_coord_name
    #     cube.add_aux_coord(coord, coord_dims)

    # def _add_time(self, cube, cubes):
    #     """Add time coordinate from other cube in cubes."""
    #     # Try to find time cube from other cubes and it to target cube
    #     for other_cube in cubes:
    #         if not other_cube.coords('time'):
    #             continue
    #         time_coord = other_cube.coord('time')
    #         cube = add_leading_dim_to_cube(cube, time_coord)
    #         return cube
    #     raise ValueError(
    #         f"Cannot add required coordinate 'time' to variable "
    #         f"'{self.vardef.short_name}', cube and other cubes in file do not "
    #         f"contain it")

    def _fix_scalar_coords(self, cube):
        """Fix scalar coordinates."""
        if 'height2m' in self.vardef.dimensions:
            add_scalar_height_coord(cube, 2.0)
        if 'height10m' in self.vardef.dimensions:
            add_scalar_height_coord(cube, 10.0)
        if 'typesi' in self.vardef.dimensions:
            add_scalar_typesi_coord(cube, 'sea_ice')

    def _fix_time(self, cube, cubes):
        """Fix time coordinate of cube."""
        # Add time coordinate if not already present
        # if not cube.coords('time'):
        #     cube = self._add_time(cube, cubes)

        # Fix metadata
        time_coord = cube.coord('time')
        time_coord.var_name = 'time'
        time_coord.standard_name = 'time'
        time_coord.long_name = 'time'

        # Add bounds if possible (not possible if cube only contains single
        # time point)
        if not time_coord.has_bounds():
            try:
                time_coord.guess_bounds()
            except ValueError:
                pass

        # if 'invalid_units' not in time_coord.attributes:
        #     return

        # # If necessary, convert invalid time units of the form "day as
        # # %Y%m%d.%f" to CF format (e.g., "days since 1850-01-01")
        # # Notes:
        # # - It might be necessary to expand this to other time formats in the
        # #   raw file.
        # # - This has not been tested with sub-daily data
        # time_format = 'day as %Y%m%d.%f'
        # t_unit = time_coord.attributes.pop('invalid_units')
        # if t_unit != time_format:
        #     raise ValueError(
        #         f"Expected time units '{time_format}' in input file, got "
        #         f"'{t_unit}'")
        # new_t_unit = cf_units.Unit('days since 1850-01-01',
        #                            calendar='proleptic_gregorian')

        # new_datetimes = [datetime.strptime(str(dt), '%Y%m%d.%f') for dt in
        #                  time_coord.points]
        # new_dt_points = date2num(np.array(new_datetimes), new_t_unit)

        # time_coord.points = new_dt_points
        # time_coord.units = new_t_unit

    def _fix_var_metadata(self, cube):
        """Fix metadata of variable."""
        if self.vardef.standard_name == '':
            cube.standard_name = None
        else:
            cube.standard_name = self.vardef.standard_name
        cube.var_name = self.vardef.short_name
        cube.long_name = self.vardef.long_name
        if cube.units != self.vardef.units:
            cube.convert_units(self.vardef.units)

    # @staticmethod
    # def _fix_height(cube, cubes):
    #     """Fix height coordinate of cube."""
    #     if cubes.extract(iris.NameConstraint(var_name='pfull')):
    #         plev_points_cube = cubes.extract_cube(
    #             iris.NameConstraint(var_name='pfull'))
    #         air_pressure_points = plev_points_cube.core_data()

    #         # Get bounds from half levels and reshape array
    #         if cubes.extract(iris.NameConstraint(var_name='phalf')):
    #             plev_bounds_cube = cubes.extract_cube(
    #                 iris.NameConstraint(var_name='phalf'))
    #             air_pressure_bounds = plev_bounds_cube.core_data()
    #             air_pressure_bounds = da.stack(
    #                 (air_pressure_bounds[:, :-1], air_pressure_bounds[:, 1:]),
    #                 axis=-1)
    #         else:
    #             air_pressure_bounds = None

    #         # Setup air pressure coordinate with correct metadata and add to
    #         # cube
    #         air_pressure_coord = iris.coords.AuxCoord(
    #             air_pressure_points,
    #             bounds=air_pressure_bounds,
    #             var_name='plev',
    #             standard_name='air_pressure',
    #             long_name='pressure',
    #             units=plev_points_cube.units,
    #             attributes={'positive': 'down'},
    #         )
    #         cube.add_aux_coord(air_pressure_coord, np.arange(cube.ndim))

    #     # Reverse entire cube along height axis so that index 0 is surface
    #     # level
    #     cube = iris.util.reverse(cube, 'height')

    #     # Fix metadata
    #     z_coord = cube.coord('height')
    #     if z_coord.units.is_convertible('m'):
    #         z_metadata = {
    #             'var_name': 'height',
    #             'standard_name': 'height',
    #             'long_name': 'height',
    #             'attributes': {'positive': 'up'},
    #         }
    #         z_coord.convert_units('m')
    #     else:
    #         z_metadata = {
    #             'var_name': 'model_level',
    #             'standard_name': None,
    #             'long_name': 'model level number',
    #             'units': 'no unit',
    #             'attributes': {'positive': 'up'},
    #             'points': np.arange(len(z_coord.points)),
    #             'bounds': None,
    #         }
    #     for (attr, val) in z_metadata.items():
    #         setattr(z_coord, attr, val)

    #     return cube

    @staticmethod
    def _fix_lat(cube, lat_name):
        """Fix latitude coordinate of cube."""
        lat = cube.coord(lat_name)
        lat.var_name = 'lat'
        lat.standard_name = 'latitude'
        lat.long_name = 'latitude'
        lat.convert_units('degrees_north')

        # Add bounds if possible (not possible if cube only contains single
        # lat point)
        if not lat.has_bounds():
            try:
                lat.guess_bounds()
            except ValueError:
                pass

    @staticmethod
    def _fix_lon(cube, lon_name):
        """Fix longitude coordinate of cube."""
        lon = cube.coord(lon_name)
        lon.var_name = 'lon'
        lon.standard_name = 'longitude'
        lon.long_name = 'longitude'
        lon.convert_units('degrees_east')

        # Add bounds if possible (not possible if cube only contains single
        # lon point)
        if not lon.has_bounds():
            try:
                lon.guess_bounds()
            except ValueError:
                pass

    def _fix_plev(self, cube):
        """Fix pressure level coordinate of cube."""
        for coord in cube.coords():
            coord_type = iris.util.guess_coord_axis(coord)
            if coord_type != 'Z':
                continue
            if not coord.units.is_convertible('Pa'):
                continue
            coord.var_name = 'plev'
            coord.standard_name = 'air_pressure'
            coord.lon_name = 'pressure'
            coord.convert_units('Pa')
            return
        raise ValueError(
            f"Cannot find requested pressure level coordinate for variable "
            f"'{self.vardef.short_name}', searched for Z coordinates with "
            f"units that are convertible to Pa")

    # @staticmethod
    # def _fix_unstructured_cell_index(cube, horizontal_idx):
    #     """Fix unstructured cell index coordinate."""
    #     index_coord = iris.coords.DimCoord(
    #         np.arange(cube.shape[horizontal_idx[0]]),
    #         var_name='i',
    #         long_name=('first spatial index for variables stored on an '
    #                    'unstructured grid'),
    #         units='1',
    #     )
    #     cube.add_dim_coord(index_coord, horizontal_idx)

    # @staticmethod
    # def _load_cubes(path):
    #     """Load cubes and ignore certain warnings."""
    #     with warnings.catch_warnings():
    #         warnings.filterwarnings(
    #             'ignore',
    #             message="Ignoring netCDF variable .* invalid units .*",
    #             category=UserWarning,
    #             module='iris',
    #         )
    #         cubes = iris.load(str(path))
    #     return cubes


class Siconc(EmacFix):
    """Fixes for ``siconc``."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        # Note: This fix is called before the AllVars() fix. The wrong var_name
        # and units (which need to be %) are fixed in a later step in
        # AllVars(). This fix here is necessary to fix the "unknown" units that
        # cannot be converted to % in AllVars().
        cube = self.get_cube(cubes)
        cube.units = '1'
        return cubes


Siconca = Siconc
