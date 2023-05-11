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
        cubes = self.add_additional_cubes(cubes)
        cube = self.get_cube(cubes)

        # Fix time
        if 'time' in self.vardef.dimensions:
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
        if 'latitude' in self.vardef.dimensions:
            lat_idx = self._fix_lat(cube)
        else:
            lat_idx = None

        # Fix longitude
        if 'longitude' in self.vardef.dimensions:
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

        # New routine to convert time of daily and hourly data
        # The string %f (fraction of day) is not a valid format string
        # for datetime.strptime, so we have to convert it ourselves
        time_str = [str(x) for x in time_coord.points]

        # First, extract date (year, month, day) from string
        # and convert it to datetime object
        year_month_day_str = pd.Series(time_str).str.extract(
            r'(\d*)\.?\d*', expand=False
        )
        year_month_day = pd.to_datetime(year_month_day_str, format='%Y%m%d')
        # Second, extract day fraction and convert it to timedelta object
        day_float_str = pd.Series(time_str).str.extract(
            r'\d*(\.\d*)', expand=False
        ).fillna('0.0')
        day_float = pd.to_timedelta(day_float_str.astype(float), unit='D')
        # Finally, add date and day fraction to get final datetime
        # and convert it to correct units
        new_datetimes = (year_month_day + day_float).dt.to_pydatetime()
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
