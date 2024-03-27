"""On-the-fly CMORizer for ICON."""

import logging
import warnings
from datetime import datetime, timedelta

import dask.array as da
import iris
import iris.util
import numpy as np
import pandas as pd
from cf_units import Unit
from iris import NameConstraint
from iris.coords import AuxCoord, DimCoord
from iris.cube import CubeList

from esmvalcore.iris_helpers import add_leading_dim_to_cube, date2num

from ._base_fixes import IconFix, NegateData

logger = logging.getLogger(__name__)


class AllVars(IconFix):
    """Fixes for all variables."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        cubes = self.add_additional_cubes(cubes)
        cube = self.get_cube(cubes)

        # Fix time
        if self.vardef.has_coord_with_standard_name('time'):
            cube = self._fix_time(cube, cubes)

        # Fix height (note: cannot use "if 'height' in self.vardef.dimensions"
        # here since the name of the z-coord varies from variable to variable)
        if cube.coords('height'):
            # In case a scalar height is required, remove it here (it is added
            # at a later stage). The step _fix_height() is designed to fix
            # non-scalar height coordinates.
            if (cube.coord('height').shape[0] == 1 and (
                    'height2m' in self.vardef.dimensions or
                    'height10m' in self.vardef.dimensions)):
                # If height is a dimensional coordinate with length 1, squeeze
                # the cube.
                # Note: iris.util.squeeze is not used here since it might
                # accidentally squeeze other dimensions.
                if cube.coords('height', dim_coords=True):
                    slices = [slice(None)] * cube.ndim
                    slices[cube.coord_dims('height')[0]] = 0
                    cube = cube[tuple(slices)]
                cube.remove_coord('height')
            else:
                cube = self._fix_height(cube, cubes)

        # Fix latitude
        if self.vardef.has_coord_with_standard_name('latitude'):
            lat_idx = self._fix_lat(cube)
        else:
            lat_idx = None

        # Fix longitude
        if self.vardef.has_coord_with_standard_name('longitude'):
            lon_idx = self._fix_lon(cube)
        else:
            lon_idx = None

        # Fix unstructured mesh of unstructured grid if present
        if self._is_unstructured_grid(lat_idx, lon_idx):
            self._fix_mesh(cube, lat_idx)

        # Fix scalar coordinates
        self.fix_scalar_coords(cube)

        # Fix metadata of variable
        self.fix_var_metadata(cube)

        return CubeList([cube])

    def _add_coord_from_grid_file(self, cube, coord_name):
        """Add coordinate from grid file to cube.

        Note
        ----
        Assumes that the input cube has a single unnamed dimension, which will
        be used as dimension for the new coordinate.

        Parameters
        ----------
        cube: iris.cube.Cube
            ICON data to which the coordinate from the grid file is added.
        coord_name: str
            Name of the coordinate to add from the grid file. Must be one of
            ``'latitude'``, ``'longitude'``.

        Raises
        ------
        ValueError
            Invalid ``coord_name`` is given; input cube does not contain a
            single unnamed dimension that can be used to add the new
            coordinate.

        """
        # The following dict maps from desired coordinate name in output file
        # (dict keys) to coordinate name in grid file (dict values)
        coord_names_mapping = {
            'latitude': 'grid_latitude',
            'longitude': 'grid_longitude',
        }
        if coord_name not in coord_names_mapping:
            raise ValueError(
                f"coord_name must be one of {list(coord_names_mapping)}, got "
                f"'{coord_name}'")
        coord_name_in_grid = coord_names_mapping[coord_name]

        # Use 'cell_area' as dummy cube to extract desired coordinates
        # Note: it might be necessary to expand this when more coord_names are
        # supported
        horizontal_grid = self.get_horizontal_grid(cube)
        grid_cube = horizontal_grid.extract_cube(
            NameConstraint(var_name='cell_area'))
        coord = grid_cube.coord(coord_name_in_grid)

        # Find index of mesh dimension (= single unnamed dimension)
        n_unnamed_dimensions = cube.ndim - len(cube.dim_coords)
        if n_unnamed_dimensions != 1:
            raise ValueError(
                f"Cannot determine coordinate dimension for coordinate "
                f"'{coord_name}', cube does not contain a single unnamed "
                f"dimension:\n{cube}")
        coord_dims = ()
        for idx in range(cube.ndim):
            if not cube.coords(dimensions=idx, dim_coords=True):
                coord_dims = (idx,)
                break

        # Adapt coordinate names so that the coordinate can be referenced with
        # 'cube.coord(coord_name)'; the exact name will be set at a later stage
        coord.standard_name = None
        coord.long_name = coord_name
        cube.add_aux_coord(coord, coord_dims)

    def _add_time(self, cube, cubes):
        """Add time coordinate from other cube in cubes."""
        # Try to find time cube from other cubes and it to target cube
        for other_cube in cubes:
            if not other_cube.coords('time'):
                continue
            time_coord = other_cube.coord('time')
            cube = add_leading_dim_to_cube(cube, time_coord)
            return cube
        raise ValueError(
            f"Cannot add required coordinate 'time' to variable "
            f"'{self.vardef.short_name}', cube and other cubes in file do not "
            f"contain it")

    def _get_z_coord(self, cubes, points_name, bounds_name=None):
        """Get z-coordinate without metadata (reversed)."""
        points_cube = iris.util.reverse(
            cubes.extract_cube(NameConstraint(var_name=points_name)),
            'height',
        )
        points = points_cube.core_data()

        # Get bounds if possible
        if bounds_name is not None:
            bounds_cube = iris.util.reverse(
                cubes.extract_cube(NameConstraint(var_name=bounds_name)),
                'height',
            )
            bounds = bounds_cube.core_data()
            bounds = da.stack(
                (bounds[..., :-1, :], bounds[..., 1:, :]), axis=-1
            )
        else:
            bounds = None

        z_coord = AuxCoord(
            points,
            bounds=bounds,
            units=points_cube.units,
        )
        return z_coord

    def _fix_height(self, cube, cubes):
        """Fix height coordinate of cube."""
        # Reverse entire cube along height axis so that index 0 is surface
        # level
        cube = iris.util.reverse(cube, 'height')

        # If possible, extract reversed air_pressure coordinate from list of
        # cubes and add it to cube
        # Note: pfull/phalf have dimensions (time, height, spatial_dim)
        if cubes.extract(NameConstraint(var_name='pfull')):
            if cubes.extract(NameConstraint(var_name='phalf')):
                phalf = 'phalf'
            else:
                phalf = None
            plev_coord = self._get_z_coord(cubes, 'pfull', bounds_name=phalf)
            self.fix_plev_metadata(cube, plev_coord)
            cube.add_aux_coord(plev_coord, np.arange(cube.ndim))

        # If possible, extract reversed altitude coordinate from list of cubes
        # and add it to cube
        # Note: zg/zghalf have dimensions (height, spatial_dim)
        if cubes.extract(NameConstraint(var_name='zg')):
            if cubes.extract(NameConstraint(var_name='zghalf')):
                zghalf = 'zghalf'
            else:
                zghalf = None
            alt_coord = self._get_z_coord(cubes, 'zg', bounds_name=zghalf)
            self.fix_alt16_metadata(cube, alt_coord)

            # Altitude coordinate only spans height and spatial dimensions (no
            # time) -> these are always the last two dimensions in the cube
            cube.add_aux_coord(alt_coord, np.arange(cube.ndim)[-2:])

        # Fix metadata
        z_coord = cube.coord('height')
        if z_coord.units.is_convertible('m'):
            self.fix_height_metadata(cube, z_coord)
        else:
            z_coord.var_name = 'model_level'
            z_coord.standard_name = None
            z_coord.long_name = 'model level number'
            z_coord.units = 'no unit'
            z_coord.attributes['positive'] = 'up'
            z_coord.points = np.arange(len(z_coord.points))
            z_coord.bounds = None

        return cube

    def _fix_lat(self, cube):
        """Fix latitude coordinate of cube."""
        lat_name = self.extra_facets.get('latitude', 'latitude')

        # Add latitude coordinate if not already present
        if not cube.coords(lat_name):
            try:
                self._add_coord_from_grid_file(cube, 'latitude')
            except Exception as exc:
                msg = "Failed to add missing latitude coordinate to cube"
                raise ValueError(msg) from exc

        # Fix metadata
        lat = self.fix_lat_metadata(cube, lat_name)

        return cube.coord_dims(lat)

    def _fix_lon(self, cube):
        """Fix longitude coordinate of cube."""
        lon_name = self.extra_facets.get('longitude', 'longitude')

        # Add longitude coordinate if not already present
        if not cube.coords(lon_name):
            try:
                self._add_coord_from_grid_file(cube, 'longitude')
            except Exception as exc:
                msg = "Failed to add missing longitude coordinate to cube"
                raise ValueError(msg) from exc

        # Fix metadata and convert to [0, 360]
        lon = self.fix_lon_metadata(cube, lon_name)
        self._set_range_in_0_360(lon)

        return cube.coord_dims(lon)

    def _fix_time(self, cube, cubes):
        """Fix time coordinate of cube."""
        # Add time coordinate if not already present
        if not cube.coords('time'):
            cube = self._add_time(cube, cubes)

        # Fix metadata
        time_coord = self.fix_time_metadata(cube)

        # If necessary, convert invalid time units of the form "day as
        # %Y%m%d.%f" to CF format (e.g., "days since 1850-01-01")
        if 'invalid_units' in time_coord.attributes:
            self._fix_invalid_time_units(time_coord)

        # ICON usually reports aggregated values at the end of the time period,
        # e.g., for monthly output, ICON reports the month February as 1 March.
        # Thus, if not disabled, shift all time points back by 1/2 of the given
        # time period.
        if self.extra_facets.get('shift_time', True):
            self._shift_time_coord(cube, time_coord)

        # If not already present, try to add bounds here. Usually bounds are
        # set in _shift_time_coord.
        self.guess_coord_bounds(cube, time_coord)

        return cube

    def _shift_time_coord(self, cube, time_coord):
        """Shift time points back by 1/2 of given time period (in-place)."""
        # Do not modify time coordinate for point measurements
        for cell_method in cube.cell_methods:
            is_point_measurement = ('time' in cell_method.coord_names and
                                    'point' in cell_method.method)
            if is_point_measurement:
                logger.debug(
                    "ICON data describes point measurements: time coordinate "
                    "will not be shifted back by 1/2 of output interval (%s)",
                    self.extra_facets['frequency'],
                )
                return

        # Remove bounds; they will be re-added later after shifting
        time_coord.bounds = None

        # For decadal, yearly and monthly data, round datetimes to closest day
        freq = self.extra_facets['frequency']
        if 'dec' in freq or 'yr' in freq or 'mon' in freq:
            time_units = time_coord.units
            time_coord.convert_units(
                Unit('days since 1850-01-01', calendar=time_units.calendar)
            )
            try:
                time_coord.points = np.around(time_coord.points)
            except ValueError as exc:
                error_msg = (
                    "Cannot shift time coordinate: Rounding to closest day "
                    "failed. Most likely you specified the wrong frequency in "
                    "the recipe (use `frequency: <your_frequency>` to fix "
                    "this). Alternatively, use `shift_time=false` in the "
                    "recipe to disable this feature"
                )
                raise ValueError(error_msg) from exc
            time_coord.convert_units(time_units)
            logger.debug(
                "Rounded ICON time coordinate to closest day for decadal, "
                "yearly and monthly data"
            )

        # Use original time points to calculate bounds (for a given point,
        # start of bounds is previous point, end of bounds is point)
        first_datetime = time_coord.units.num2date(time_coord.points[0])
        previous_time_point = time_coord.units.date2num(
            self._get_previous_timestep(first_datetime)
        )
        extended_time_points = np.concatenate(
            ([previous_time_point], time_coord.points)
        )
        time_coord.points = (
            np.convolve(extended_time_points, np.ones(2), 'valid') / 2.0
        )  # running mean with window length 2
        time_coord.bounds = np.stack(
            (extended_time_points[:-1], extended_time_points[1:]), axis=-1
        )
        logger.debug(
            "Shifted ICON time coordinate back by 1/2 of output interval (%s)",
            self.extra_facets['frequency'],
        )

    def _get_previous_timestep(self, datetime_point):
        """Get previous time step."""
        freq = self.extra_facets['frequency']
        year = datetime_point.year
        month = datetime_point.month

        # Invalid input
        invalid_freq_error_msg = (
            f"Cannot shift time coordinate: failed to determine previous time "
            f"step for frequency '{freq}'. Use `shift_time=false` in the "
            f"recipe to disable this feature"
        )
        if 'fx' in freq or 'subhr' in freq:
            raise ValueError(invalid_freq_error_msg)

        # For decadal, yearly and monthly data, the points needs to be the
        # first of the month 00:00:00
        if 'dec' in freq or 'yr' in freq or 'mon' in freq:
            if datetime_point != datetime(year, month, 1):
                raise ValueError(
                    f"Cannot shift time coordinate: expected first of the "
                    f"month at 00:00:00 for decadal, yearly and monthly data, "
                    f"got {datetime_point}. Use `shift_time=false` in the "
                    f"recipe to disable this feature"
                )

        # Decadal data
        if 'dec' in freq:
            return datetime_point.replace(year=year - 10)

        # Yearly data
        if 'yr' in freq:
            return datetime_point.replace(year=year - 1)

        # Monthly data
        if 'mon' in freq:
            new_month = (month - 2) % 12 + 1
            new_year = year + (month - 2) // 12
            return datetime_point.replace(year=new_year, month=new_month)

        # Daily data
        if 'day' in freq:
            return datetime_point - timedelta(days=1)

        # Hourly data
        if 'hr' in freq:
            (n_hours, _, _) = freq.partition('hr')
            if not n_hours:
                n_hours = 1
            return datetime_point - timedelta(hours=int(n_hours))

        # Unknown input
        raise ValueError(invalid_freq_error_msg)

    def _fix_mesh(self, cube, mesh_idx):
        """Fix mesh."""
        # Remove any already-present dimensional coordinate describing the mesh
        # dimension
        if cube.coords(dimensions=mesh_idx, dim_coords=True):
            cube.remove_coord(cube.coord(dimensions=mesh_idx, dim_coords=True))

        # Add dimensional coordinate that describes the mesh dimension
        index_coord = DimCoord(
            np.arange(cube.shape[mesh_idx[0]]),
            var_name='i',
            long_name=('first spatial index for variables stored on an '
                       'unstructured grid'),
            units='1',
        )
        cube.add_dim_coord(index_coord, mesh_idx)

        # If desired, get mesh and replace the original latitude and longitude
        # coordinates with their new mesh versions
        if self.extra_facets.get('ugrid', True):
            mesh = self.get_mesh(cube)
            cube.remove_coord('latitude')
            cube.remove_coord('longitude')
            for mesh_coord in mesh.to_MeshCoords('face'):
                cube.add_aux_coord(mesh_coord, mesh_idx)

    @staticmethod
    def _is_unstructured_grid(lat_idx, lon_idx):
        """Check if data is defined on an unstructured grid."""
        # If either latitude or longitude are not present (i.e., the
        # corresponding index is None), no unstructured grid is present
        if lat_idx is None:
            return False
        if lon_idx is None:
            return False

        # If latitude and longitude do not share their dimensions, no
        # unstructured grid is present
        if lat_idx != lon_idx:
            return False

        # If latitude and longitude are multi-dimensional (e.g., curvilinear
        # grid), no unstructured grid is present
        if len(lat_idx) != 1:
            return False

        return True

    @staticmethod
    def _fix_invalid_time_units(time_coord):
        """Fix invalid time units (in-place)."""
        # ICON data usually has no time bounds. To be 100% sure, we remove the
        # bounds here (they will be added at a later stage).
        time_coord.bounds = None
        time_format = 'day as %Y%m%d.%f'
        t_unit = time_coord.attributes.pop('invalid_units')
        if t_unit != time_format:
            raise ValueError(
                f"Expected time units '{time_format}' in input file, got "
                f"'{t_unit}'"
            )
        new_t_units = Unit(
            'days since 1850-01-01', calendar='proleptic_gregorian'
        )

        # New routine to convert time of daily and hourly data. The string %f
        # (fraction of day) is not a valid format string for datetime.strptime,
        # so we have to convert it ourselves.
        time_str = pd.Series(time_coord.points, dtype=str)

        # First, extract date (year, month, day) from string and convert it to
        # datetime object
        year_month_day_str = time_str.str.extract(r'(\d*)\.?\d*', expand=False)
        year_month_day = pd.to_datetime(year_month_day_str, format='%Y%m%d')

        # Second, extract day fraction and convert it to timedelta object
        day_float_str = time_str.str.extract(
            r'\d*(\.\d*)', expand=False
        ).fillna('0.0')
        day_float = pd.to_timedelta(day_float_str.astype(float), unit='D')

        # Finally, add date and day fraction to get final datetime and convert
        # it to correct units. Note: we also round to next second, otherwise
        # this results in times that are off by 1s (e.g., 13:59:59 instead of
        # 14:00:00).
        rounded_datetimes = (year_month_day + day_float).round('s')
        with warnings.catch_warnings():
            # We already fixed the deprecated code as recommended in the
            # warning, but it still shows up -> ignore it
            warnings.filterwarnings(
                'ignore',
                message="The behavior of DatetimeProperties.to_pydatetime .*",
                category=FutureWarning,
            )
            new_datetimes = np.array(rounded_datetimes.dt.to_pydatetime())
        new_dt_points = date2num(np.array(new_datetimes), new_t_units)

        # Modify time coordinate in place
        time_coord.points = new_dt_points
        time_coord.units = new_t_units


class Clwvi(IconFix):
    """Fixes for ``clwvi``."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        cube = (
            self.get_cube(cubes, var_name='cllvi') +
            self.get_cube(cubes, var_name='clivi')
        )
        cube.var_name = self.vardef.short_name
        return CubeList([cube])


class Rtmt(IconFix):
    """Fixes for ``rtmt``."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        cube = (
            self.get_cube(cubes, var_name='rsdt') -
            self.get_cube(cubes, var_name='rsut') -
            self.get_cube(cubes, var_name='rlut')
        )
        cube.var_name = self.vardef.short_name
        return CubeList([cube])


Hfls = NegateData


Hfss = NegateData


Rtnt = Rtmt
