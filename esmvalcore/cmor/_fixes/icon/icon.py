"""On-the-fly CMORizer for ICON."""

import logging
from datetime import datetime

import cf_units
import dask.array as da
import iris
import iris.util
import numpy as np
from iris import NameConstraint
from iris.coords import AuxCoord, DimCoord
from iris.cube import CubeList

from esmvalcore.iris_helpers import add_leading_dim_to_cube, date2num

from ..shared import add_scalar_height_coord, add_scalar_typesi_coord
from ._base_fixes import IconFix

logger = logging.getLogger(__name__)


class AllVars(IconFix):
    """Fixes for all variables."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
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

        # Fix cell index for unstructured grid if necessary
        if self._cell_index_needs_fixing(lat_idx, lon_idx):
            self._fix_unstructured_cell_index(cube, lat_idx)

        # Fix scalar coordinates
        self._fix_scalar_coords(cube)

        # Fix metadata of variable
        self._fix_var_metadata(cube)

        return CubeList([cube])

    def _add_coord_from_grid_file(self, cube, coord_name,
                                  target_coord_long_name):
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
            Name of the coordinate in the grid file. Must be one of
            ``'grid_latitude'``, ``'grid_longitude'``.
        target_coord_long_name: str
            Long name that is assigned to the newly added coordinate.

        Raises
        ------
        ValueError
            Invalid ``coord_name`` is given; input cube does not contain a
            single unnamed dimension that can be used to add the new
            coordinate.

        """
        allowed_coord_names = ('grid_latitude', 'grid_longitude')
        if coord_name not in allowed_coord_names:
            raise ValueError(
                f"coord_name must be one of {allowed_coord_names}, got "
                f"'{coord_name}'")
        horizontal_grid = self.get_horizontal_grid(cube)

        # Use 'cell_area' as dummy cube to extract coordinates
        # Note: it might be necessary to expand this when more coord_names are
        # supported
        grid_cube = horizontal_grid.extract_cube(
            NameConstraint(var_name='cell_area'))
        coord = grid_cube.coord(coord_name)

        # Find index of horizontal coordinate (= single unnamed dimension)
        n_unnamed_dimensions = cube.ndim - len(cube.dim_coords)
        if n_unnamed_dimensions != 1:
            raise ValueError(
                f"Cannot determine coordinate dimension for coordinate "
                f"'{target_coord_long_name}', cube does not contain a single "
                f"unnamed dimension:\n{cube}")
        coord_dims = ()
        for idx in range(cube.ndim):
            if not cube.coords(dimensions=idx, dim_coords=True):
                coord_dims = (idx,)
                break

        coord.standard_name = None
        coord.long_name = target_coord_long_name
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

    def _fix_lat(self, cube):
        """Fix latitude coordinate of cube."""
        lat_name = self.extra_facets.get('latitude', 'latitude')

        # Add latitude coordinate if not already present
        if not cube.coords(lat_name):
            try:
                self._add_coord_from_grid_file(cube, 'grid_latitude', lat_name)
            except Exception as exc:
                msg = "Failed to add missing latitude coordinate to cube"
                raise ValueError(msg) from exc

        # Fix metadata
        lat = cube.coord(lat_name)
        lat.var_name = 'lat'
        lat.standard_name = 'latitude'
        lat.long_name = 'latitude'
        lat.convert_units('degrees_north')

        return cube.coord_dims(lat)

    def _fix_lon(self, cube):
        """Fix longitude coordinate of cube."""
        lon_name = self.extra_facets.get('longitude', 'longitude')

        # Add longitude coordinate if not already present
        if not cube.coords(lon_name):
            try:
                self._add_coord_from_grid_file(
                    cube, 'grid_longitude', lon_name)
            except Exception as exc:
                msg = "Failed to add missing longitude coordinate to cube"
                raise ValueError(msg) from exc

        # Fix metadata
        lon = cube.coord(lon_name)
        lon.var_name = 'lon'
        lon.standard_name = 'longitude'
        lon.long_name = 'longitude'
        lon.convert_units('degrees_east')

        return cube.coord_dims(lon)

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
        if not cube.coords('time'):
            cube = self._add_time(cube, cubes)

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

        if 'invalid_units' not in time_coord.attributes:
            return cube

        # If necessary, convert invalid time units of the form "day as
        # %Y%m%d.%f" to CF format (e.g., "days since 1850-01-01")
        # Notes:
        # - It might be necessary to expand this to other time formats in the
        #   raw file.
        # - This has not been tested with sub-daily data
        time_format = 'day as %Y%m%d.%f'
        t_unit = time_coord.attributes.pop('invalid_units')
        if t_unit != time_format:
            raise ValueError(
                f"Expected time units '{time_format}' in input file, got "
                f"'{t_unit}'")
        new_t_unit = cf_units.Unit('days since 1850-01-01',
                                   calendar='proleptic_gregorian')

        new_datetimes = [datetime.strptime(str(dt), '%Y%m%d.%f') for dt in
                         time_coord.points]
        new_dt_points = date2num(np.array(new_datetimes), new_t_unit)

        time_coord.points = new_dt_points
        time_coord.units = new_t_unit

        return cube

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
        if self.vardef.positive != '':
            cube.attributes['positive'] = self.vardef.positive

    @staticmethod
    def _cell_index_needs_fixing(lat_idx, lon_idx):
        """Check if cell index coordinate of unstructured grid needs fixing."""
        # If either latitude or longitude are not present (i.e., the
        # corresponding index is None), no fix is necessary
        if lat_idx is None:
            return False
        if lon_idx is None:
            return False

        # If latitude and longitude do not share their dimensions, no fix is
        # necessary
        if lat_idx != lon_idx:
            return False

        # If latitude and longitude are multi-dimensional (i.e., curvilinear
        # instead of unstructured grid is given), no fix is necessary
        if len(lat_idx) != 1:
            return False

        return True

    @staticmethod
    def _fix_height(cube, cubes):
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
            z_metadata = {
                'var_name': 'height',
                'standard_name': 'height',
                'long_name': 'height',
                'attributes': {'positive': 'up'},
            }
            z_coord.convert_units('m')
        else:
            z_metadata = {
                'var_name': 'model_level',
                'standard_name': None,
                'long_name': 'model level number',
                'units': 'no unit',
                'attributes': {'positive': 'up'},
                'points': np.arange(len(z_coord.points)),
                'bounds': None,
            }
        for (attr, val) in z_metadata.items():
            setattr(z_coord, attr, val)

        return cube

    @staticmethod
    def _fix_unstructured_cell_index(cube, horizontal_idx):
        """Fix unstructured cell index coordinate."""
        if cube.coords(dimensions=horizontal_idx, dim_coords=True):
            cube.remove_coord(cube.coord(dimensions=horizontal_idx,
                                         dim_coords=True))
        index_coord = DimCoord(
            np.arange(cube.shape[horizontal_idx[0]]),
            var_name='i',
            long_name=('first spatial index for variables stored on an '
                       'unstructured grid'),
            units='1',
        )
        cube.add_dim_coord(index_coord, horizontal_idx)


class Siconc(IconFix):
    """Fixes for ``siconc``."""

    def fix_metadata(self, cubes):
        """Fix metadata.

        Note
        ----
        This fix is called before the AllVars() fix. The wrong var_name and
        units (which need to be %) are fixed in a later step in AllVars(). This
        fix here is necessary to fix the "unknown" units that cannot be
        converted to % in AllVars().

        """
        cube = self.get_cube(cubes)
        cube.units = '1'
        return cubes


Siconca = Siconc
