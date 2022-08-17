"""Common fix operations for native datasets."""

import logging
from typing import Dict

from iris import NameConstraint

from ..fix import Fix
from .shared import (
    add_scalar_height_coord,
    add_scalar_lambda550nm_coord,
    add_scalar_typesi_coord,
)

logger = logging.getLogger(__name__)


class NativeDatasetFix(Fix):
    """Common fix operations for native datasets."""

    # Dictionary to map invalid units in the data to valid entries
    INVALID_UNITS: Dict[str, str] = {}

    def fix_scalar_coords(self, cube):
        """Add missing scalar coordinate to cube (in-place).

        Parameters
        ----------
        cube: iris.cube.Cube
            Input cube for which missing scalar coordinates will be added
            (in-place).

        """
        if 'height2m' in self.vardef.dimensions:
            add_scalar_height_coord(cube, 2.0)
        if 'height10m' in self.vardef.dimensions:
            add_scalar_height_coord(cube, 10.0)
        if 'lambda550nm' in self.vardef.dimensions:
            add_scalar_lambda550nm_coord(cube)
        if 'typesi' in self.vardef.dimensions:
            add_scalar_typesi_coord(cube, 'sea_ice')

    def fix_var_metadata(self, cube):
        """Fix variable metadata of cube (in-place).

        Parameters
        ----------
        cube: iris.cube.Cube
            Input cube whose metadata is changed in-place.

        """
        # Fix names
        if self.vardef.standard_name == '':
            cube.standard_name = None
        else:
            cube.standard_name = self.vardef.standard_name
        cube.var_name = self.vardef.short_name
        cube.long_name = self.vardef.long_name

        # Fix units (also handles invalid units in the input files)
        if 'invalid_units' in cube.attributes:
            invalid_units = cube.attributes.pop('invalid_units')
            new_units = self.INVALID_UNITS.get(
                invalid_units,
                invalid_units.replace('**', '^'),
            )
            try:
                cube.units = new_units
            except ValueError as exc:
                raise ValueError(
                    f"Failed to fix invalid units '{invalid_units}' for "
                    f"variable '{self.vardef.short_name}'") from exc
        cube.convert_units(self.vardef.units)

        # Fix attributes
        if self.vardef.positive != '':
            cube.attributes['positive'] = self.vardef.positive

    def get_cube(self, cubes, var_name=None):
        """Extract single cube from :class:`iris.cube.CubeList`.

        Parameters
        ----------
        cubes: iris.cube.CubeList
            Input cubes.
        var_name: str, optional
            If given, use this `var_name` to extract the desired cube. If not
            given, use the `raw_name` given in extra_facets (if possible) or
            the `short_name` of the variable to extract the desired cube.

        Returns
        -------
        iris.cube.Cube
            Desired cube.

        Raises
        ------
        ValueError
            Desired variable is not available in the input cubes.

        """
        if var_name is None:
            var_name = self.extra_facets.get('raw_name',
                                             self.vardef.short_name)
        if not cubes.extract(NameConstraint(var_name=var_name)):
            raise ValueError(
                f"Variable '{var_name}' used to extract "
                f"'{self.vardef.short_name}' is not available in input file")
        return cubes.extract_cube(NameConstraint(var_name=var_name))

    def fix_regular_time(self, cube, coord=None, guess_bounds=True):
        """Fix regular time coordinate.

        Parameters
        ----------
        cube: iris.cube.Cube
            Input cube.
        coord: str or iris.coords.Coord or None, optional (default: None)
            Coordinate for which metadata will be fixed in-place. If ``None``,
            assume the coordinate's name is `time`.
        guess_bounds: bool, optional (default: True)
            If ``True``, try to guess bounds. If ``False``, do not try to guess
            bounds.

        """
        if 'time' not in self.vardef.dimensions:
            return
        coord = self.fix_time_metadata(cube, coord)
        if guess_bounds:
            self.guess_coord_bounds(cube, coord)

    def fix_regular_lat(self, cube, coord=None, guess_bounds=True):
        """Fix regular latitude coordinate.

        Parameters
        ----------
        cube: iris.cube.Cube
            Input cube.
        coord: str or iris.coords.Coord or None, optional (default: None)
            Coordinate for which metadata will be fixed in-place. If ``None``,
            assume the coordinate's name is `latitude`.
        guess_bounds: bool, optional (default: True)
            If ``True``, try to guess bounds. If ``False``, do not try to guess
            bounds.

        """
        if 'latitude' not in self.vardef.dimensions:
            return
        coord = self.fix_lat_metadata(cube, coord)
        if guess_bounds:
            self.guess_coord_bounds(cube, coord)

    def fix_regular_lon(self, cube, coord=None, guess_bounds=True):
        """Fix regular longitude coordinate.

        Parameters
        ----------
        cube: iris.cube.Cube
            Input cube.
        coord: str or iris.coords.Coord or None, optional (default: None)
            Coordinate for which metadata will be fixed in-place. If ``None``,
            assume the coordinate's name is `longitude`.
        guess_bounds: bool, optional (default: True)
            If ``True``, try to guess bounds. If ``False``, do not try to guess
            bounds.

        """
        if 'longitude' not in self.vardef.dimensions:
            return
        coord = self.fix_lon_metadata(cube, coord)
        if guess_bounds:
            self.guess_coord_bounds(cube, coord)

    @staticmethod
    def guess_coord_bounds(cube, coord):
        """Guess bounds for a coordinate (in-place).

        Note
        ----
        Bounds will not be guessed if bounds are already present or if only one
        point is available (no exception is raised in these case).

        Parameters
        ----------
        cube: iris.cube.Cube
            Input cube.
        coord: str or iris.coords.Coord or None, optional (default: None)
            Coordinate for which bounds will be guessed in-place.

        Returns
        -------
        iris.coords.AuxCoord or iris.coords.DimCoord
            Coordinate with bounds. The coordinate is altered in-place; it is
            just returned out of convenience for easy access.

        """
        if isinstance(coord, str):
            coord = cube.coord(coord)
        if not coord.has_bounds():
            try:
                coord.guess_bounds()
            except ValueError:  # Coord has only 1 point
                pass
        return coord

    @staticmethod
    def fix_time_metadata(cube, coord=None):
        """Fix metadata of time coordinate (in-place).

        Parameters
        ----------
        cube: iris.cube.Cube
            Input cube.
        coord: str or iris.coords.Coord or None, optional (default: None)
            Coordinate for which metadata will be fixed in-place. If ``None``,
            assume the coordinate's name is `time`.

        Returns
        -------
        iris.coords.AuxCoord or iris.coords.DimCoord
            Fixed time coordinate. The coordinate is altered in-place; it is
            just returned out of convenience for easy access.

        """
        if coord is None:
            coord = cube.coord('time')
        elif isinstance(coord, str):
            coord = cube.coord(coord)
        coord.var_name = 'time'
        coord.standard_name = 'time'
        coord.long_name = 'time'
        return coord

    @staticmethod
    def fix_height_metadata(cube, coord=None):
        """Fix metadata of height coordinate (in-place).

        Parameters
        ----------
        cube: iris.cube.Cube
            Input cube.
        coord: str or iris.coords.Coord or None, optional (default: None)
            Coordinate for which metadata will be fixed in-place. If ``None``,
            assume the coordinate's name is `height`.

        Returns
        -------
        iris.coords.AuxCoord or iris.coords.DimCoord
            Fixed height coordinate. The coordinate is altered in-place; it is
            just returned out of convenience for easy access.

        """
        if coord is None:
            coord = cube.coord('height')
        elif isinstance(coord, str):
            coord = cube.coord(coord)
        coord.var_name = 'height'
        coord.standard_name = 'height'
        coord.long_name = 'height'
        coord.convert_units('m')
        coord.attributes['positive'] = 'up'
        return coord

    @staticmethod
    def fix_plev_metadata(cube, coord=None):
        """Fix metadata of air_pressure coordinate (in-place).

        Parameters
        ----------
        cube: iris.cube.Cube
            Input cube.
        coord: str or iris.coords.Coord or None, optional (default: None)
            Coordinate for which metadata will be fixed in-place. If ``None``,
            assume the coordinate's name is `air_pressure`.

        Returns
        -------
        iris.coords.AuxCoord or iris.coords.DimCoord
            Fixed air_pressure coordinate. The coordinate is altered in-place;
            it is just returned out of convenience for easy access.

        """
        if coord is None:
            coord = cube.coord('air_pressure')
        elif isinstance(coord, str):
            coord = cube.coord(coord)
        coord.var_name = 'plev'
        coord.standard_name = 'air_pressure'
        coord.long_name = 'pressure'
        coord.convert_units('Pa')
        coord.attributes['positive'] = 'down'
        return coord

    @staticmethod
    def fix_lat_metadata(cube, coord=None):
        """Fix metadata of latitude coordinate (in-place).

        Parameters
        ----------
        cube: iris.cube.Cube
            Input cube.
        coord: str or iris.coords.Coord or None, optional (default: None)
            Coordinate for which metadata will be fixed in-place. If ``None``,
            assume the coordinate's name is `latitude`.

        Returns
        -------
        iris.coords.AuxCoord or iris.coords.DimCoord
            Fixed latitude coordinate. The coordinate is altered in-place; it
            is just returned out of convenience for easy access.

        """
        if coord is None:
            coord = cube.coord('latitude')
        elif isinstance(coord, str):
            coord = cube.coord(coord)
        coord.var_name = 'lat'
        coord.standard_name = 'latitude'
        coord.long_name = 'latitude'
        coord.convert_units('degrees_north')
        return coord

    @staticmethod
    def fix_lon_metadata(cube, coord=None):
        """Fix metadata of longitude coordinate (in-place).

        Parameters
        ----------
        cube: iris.cube.Cube
            Input cube.
        coord: str or iris.coords.Coord or None, optional (default: None)
            Coordinate for which metadata will be fixed in-place. If ``None``,
            assume the coordinate's name is `longitude`.

        Returns
        -------
        iris.coords.AuxCoord or iris.coords.DimCoord
            Fixed longitude coordinate. The coordinate is altered in-place; it
            is just returned out of convenience for easy access.

        """
        if coord is None:
            coord = cube.coord('longitude')
        elif isinstance(coord, str):
            coord = cube.coord(coord)
        coord.var_name = 'lon'
        coord.standard_name = 'longitude'
        coord.long_name = 'longitude'
        coord.convert_units('degrees_east')
        return coord
