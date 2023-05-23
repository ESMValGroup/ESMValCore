"""On-the-fly CMORizer for ICON."""

import logging

import cf_units
import dask.array as da
import iris
import iris.util
import numpy as np
import pandas as pd
from iris import NameConstraint
from iris.coords import AuxCoord, DimCoord
from iris.cube import CubeList

from esmvalcore.iris_helpers import add_leading_dim_to_cube, date2num

from ._base_fixes import IconFix, SetUnitsTo1

logger = logging.getLogger(__name__)


class AllVars(IconFix):
    """Fixes for all variables."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
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

    def _fix_height(self, cube, cubes):
        """Fix height coordinate of cube."""
        # Reverse entire cube along height axis so that index 0 is surface
        # level
        cube = iris.util.reverse(cube, 'height')

        # Add air_pressure coordinate if possible
        # (make sure to also reverse pressure cubes)
        if cubes.extract(NameConstraint(var_name='pfull')):
            plev_points_cube = iris.util.reverse(
                cubes.extract_cube(NameConstraint(var_name='pfull')),
                'height',
            )
            air_pressure_points = plev_points_cube.core_data()

            # Get bounds from half levels and reshape array
            if cubes.extract(NameConstraint(var_name='phalf')):
                plev_bounds_cube = iris.util.reverse(
                    cubes.extract_cube(NameConstraint(var_name='phalf')),
                    'height',
                )
                air_pressure_bounds = plev_bounds_cube.core_data()
                air_pressure_bounds = da.stack(
                    (air_pressure_bounds[:, :-1], air_pressure_bounds[:, 1:]),
                    axis=-1)
            else:
                air_pressure_bounds = None

            # Setup air pressure coordinate with correct metadata and add to
            # cube
            air_pressure_coord = AuxCoord(
                air_pressure_points,
                bounds=air_pressure_bounds,
                var_name='plev',
                standard_name='air_pressure',
                long_name='pressure',
                units=plev_points_cube.units,
                attributes={'positive': 'down'},
            )
            cube.add_aux_coord(air_pressure_coord, np.arange(cube.ndim))

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
        if 'invalid_units' not in time_coord.attributes:
            self.guess_coord_bounds(cube, time_coord)
            return cube

        # If necessary, convert invalid time units of the form "day as
        # %Y%m%d.%f" to CF format (e.g., "days since 1850-01-01")
        # ICON data has no time bounds, let's make sure we remove the bounds
        # here (they will be added after converting the time points to the
        # correct units)
        time_coord.bounds = None
        time_format = 'day as %Y%m%d.%f'
        t_unit = time_coord.attributes.pop('invalid_units')
        if t_unit != time_format:
            raise ValueError(
                f"Expected time units '{time_format}' in input file, got "
                f"'{t_unit}'")
        new_t_unit = cf_units.Unit('days since 1850-01-01',
                                   calendar='proleptic_gregorian')

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
        new_datetimes = (year_month_day + day_float).round(
            'S'
        ).dt.to_pydatetime()
        new_dt_points = date2num(np.array(new_datetimes), new_t_unit)

        time_coord.points = new_dt_points
        time_coord.units = new_t_unit
        self.guess_coord_bounds(cube, time_coord)

        return cube

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


Hur = SetUnitsTo1


Siconc = SetUnitsTo1


Siconca = SetUnitsTo1
