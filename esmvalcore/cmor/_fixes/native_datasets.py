"""Common fix operations for native datasets."""

from __future__ import annotations

import logging
from contextlib import suppress
from typing import TYPE_CHECKING, ClassVar

from iris import NameConstraint

from esmvalcore.cmor.fix import Fix
from esmvalcore.iris_helpers import safe_convert_units

from .shared import (
    add_scalar_height_coord,
    add_scalar_lambda550nm_coord,
    add_scalar_typesi_coord,
)

if TYPE_CHECKING:
    from iris.coords import Coord
    from iris.cube import Cube, CubeList

logger = logging.getLogger(__name__)


class NativeDatasetFix(Fix):
    """Common fix operations for native datasets."""

    # Dictionary to map invalid units in the data to valid entries
    INVALID_UNITS: ClassVar[dict[str, str]] = {}

    def fix_scalar_coords(self, cube: Cube) -> None:
        """Add missing scalar coordinate to cube (in-place).

        Parameters
        ----------
        cube:
            Input cube for which missing scalar coordinates will be added
            (in-place).

        """
        if "height2m" in self.vardef.dimensions:
            add_scalar_height_coord(cube, 2.0)
        if "height10m" in self.vardef.dimensions:
            add_scalar_height_coord(cube, 10.0)
        if "lambda550nm" in self.vardef.dimensions:
            add_scalar_lambda550nm_coord(cube)
        if "typesi" in self.vardef.dimensions:
            add_scalar_typesi_coord(cube, "sea_ice")

    def fix_var_metadata(self, cube: Cube) -> None:
        """Fix variable metadata of cube (in-place).

        Parameters
        ----------
        cube:
            Input cube whose metadata is changed in-place.

        """
        # Fix names
        if self.vardef.standard_name == "":
            cube.standard_name = None
        else:
            cube.standard_name = self.vardef.standard_name
        cube.var_name = self.vardef.short_name
        cube.long_name = self.vardef.long_name

        # Fix units
        # (1) raw_units set in recipe or extra_facets
        if "raw_units" in self.extra_facets:
            cube.units = self.extra_facets["raw_units"]
            cube.attributes.pop("invalid_units", None)

        # (2) Try to handle other invalid units in the input files
        if "invalid_units" in cube.attributes:
            invalid_units = cube.attributes.pop("invalid_units")
            new_units = self.INVALID_UNITS.get(
                invalid_units,
                invalid_units.replace("**", "^"),
            )
            try:
                cube.units = new_units
            except ValueError as exc:
                msg = (
                    f"Failed to fix invalid units '{invalid_units}' for "
                    f"variable '{self.vardef.short_name}'"
                )
                raise ValueError(
                    msg,
                ) from exc
        safe_convert_units(cube, self.vardef.units)

        # Fix attributes
        if self.vardef.positive != "":
            cube.attributes["positive"] = self.vardef.positive

    def get_cube(
        self,
        cubes: CubeList,
        var_name: str | None = None,
    ) -> Cube:
        """Extract single cube from :class:`iris.cube.CubeList`.

        Parameters
        ----------
        cubes:
            Input cubes.
        var_name:
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
            var_name = self.extra_facets.get(
                "raw_name",
                self.vardef.short_name,
            )
        if not cubes.extract(NameConstraint(var_name=var_name)):
            msg = (
                f"Variable '{var_name}' used to extract "
                f"'{self.vardef.short_name}' is not available in input file"
            )
            raise ValueError(
                msg,
            )
        return cubes.extract_cube(NameConstraint(var_name=var_name))

    def fix_regular_time(
        self,
        cube: Cube,
        coord: str | Coord | None = None,
        guess_bounds: bool = True,
    ) -> None:
        """Fix regular time coordinate.

        Parameters
        ----------
        cube:
            Input cube.
        coord:
            Coordinate for which metadata will be fixed in-place. If ``None``,
            assume the coordinate's name is `time`.
        guess_bounds:
            If ``True``, try to guess bounds. If ``False``, do not try to guess
            bounds.

        """
        if not self.vardef.has_coord_with_standard_name("time"):
            return
        coord = self.fix_time_metadata(cube, coord)
        if guess_bounds:
            self.guess_coord_bounds(cube, coord)

    def fix_regular_lat(
        self,
        cube: Cube,
        coord: str | Coord | None = None,
        guess_bounds: bool = True,
    ) -> None:
        """Fix regular latitude coordinate.

        Parameters
        ----------
        cube:
            Input cube.
        coord:
            Coordinate for which metadata will be fixed in-place. If ``None``,
            assume the coordinate's name is `latitude`.
        guess_bounds:
            If ``True``, try to guess bounds. If ``False``, do not try to guess
            bounds.

        """
        if not self.vardef.has_coord_with_standard_name("latitude"):
            return
        coord = self.fix_lat_metadata(cube, coord)
        if guess_bounds:
            self.guess_coord_bounds(cube, coord)

    def fix_regular_lon(
        self,
        cube: Cube,
        coord: str | Coord | None = None,
        guess_bounds: bool = True,
    ) -> None:
        """Fix regular longitude coordinate.

        Parameters
        ----------
        cube:
            Input cube.
        coord:
            Coordinate for which metadata will be fixed in-place. If ``None``,
            assume the coordinate's name is `longitude`.
        guess_bounds:
            If ``True``, try to guess bounds. If ``False``, do not try to guess
            bounds.

        """
        if not self.vardef.has_coord_with_standard_name("longitude"):
            return
        coord = self.fix_lon_metadata(cube, coord)
        if guess_bounds:
            self.guess_coord_bounds(cube, coord)

    @staticmethod
    def guess_coord_bounds(cube: Cube, coord: Coord) -> Coord:
        """Guess bounds for a coordinate (in-place).

        Note
        ----
        Bounds will not be guessed if bounds are already present or if only one
        point is available (no exception is raised in these case).

        Parameters
        ----------
        cube:
            Input cube.
        coord:
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
            with suppress(ValueError):
                coord.guess_bounds()
        return coord

    @staticmethod
    def fix_time_metadata(
        cube: Cube,
        coord: str | Coord | None = None,
    ) -> Coord:
        """Fix metadata of time coordinate (in-place).

        Parameters
        ----------
        cube:
            Input cube.
        coord:
            Coordinate for which metadata will be fixed in-place. If ``None``,
            assume the coordinate's name is `time`.

        Returns
        -------
        iris.coords.AuxCoord or iris.coords.DimCoord
            Fixed time coordinate. The coordinate is altered in-place; it is
            just returned out of convenience for easy access.

        """
        if coord is None:
            coord = cube.coord("time")
        elif isinstance(coord, str):
            coord = cube.coord(coord)
        coord.var_name = "time"  # type: ignore
        coord.standard_name = "time"  # type: ignore
        coord.long_name = "time"  # type: ignore
        return coord

    @staticmethod
    def fix_alt16_metadata(
        cube: Cube,
        coord: str | Coord | None = None,
    ) -> Coord:
        """Fix metadata of alt16 coordinate (in-place).

        Parameters
        ----------
        cube:
            Input cube.
        coord:
            Coordinate for which metadata will be fixed in-place. If ``None``,
            assume the coordinate's name is `altitude`.

        Returns
        -------
        iris.coords.AuxCoord or iris.coords.DimCoord
            Fixed altitude coordinate. The coordinate is altered in-place; it
            is just returned out of convenience for easy access.

        """
        if coord is None:
            coord = cube.coord("altitude")
        elif isinstance(coord, str):
            coord = cube.coord(coord)
        coord.var_name = "alt16"  # type: ignore
        coord.standard_name = "altitude"  # type: ignore
        coord.long_name = "altitude"  # type: ignore
        coord.convert_units("m")  # type: ignore
        coord.attributes["positive"] = "up"  # type: ignore
        return coord

    @staticmethod
    def fix_height_metadata(
        cube: Cube,
        coord: str | Coord | None = None,
    ) -> Coord:
        """Fix metadata of height coordinate (in-place).

        Parameters
        ----------
        cube:
            Input cube.
        coord:
            Coordinate for which metadata will be fixed in-place. If ``None``,
            assume the coordinate's name is `height`.

        Returns
        -------
        iris.coords.AuxCoord or iris.coords.DimCoord
            Fixed height coordinate. The coordinate is altered in-place; it is
            just returned out of convenience for easy access.

        """
        if coord is None:
            coord = cube.coord("height")
        elif isinstance(coord, str):
            coord = cube.coord(coord)
        coord.var_name = "height"  # type: ignore
        coord.standard_name = "height"  # type: ignore
        coord.long_name = "height"  # type: ignore
        coord.convert_units("m")  # type: ignore
        coord.attributes["positive"] = "up"  # type: ignore
        return coord

    @staticmethod
    def fix_depth_coord_metadata(
        cube: Cube,
        coord: str | Coord | None = None,
    ) -> Coord:
        """Fix metadata of depth_coord coordinate (in-place).

        Parameters
        ----------
        cube:
            Input cube.
        coord:
            Coordinate for which metadata will be fixed in-place. If ``None``,
            assume the coordinate's name is `depth`.

        Returns
        -------
        iris.coords.AuxCoord or iris.coords.DimCoord
            Fixed depth coordinate. The coordinate is altered in-place; it is
            just returned out of convenience for easy access.

        """
        if coord is None:
            coord = cube.coord("depth")
        elif isinstance(coord, str):
            coord = cube.coord(coord)
        coord.var_name = "lev"  # type: ignore
        coord.standard_name = "depth"  # type: ignore
        coord.long_name = "ocean depth coordinate"  # type: ignore
        coord.convert_units("m")  # type: ignore
        coord.attributes["positive"] = "down"  # type: ignore
        return coord

    @staticmethod
    def fix_plev_metadata(
        cube: Cube,
        coord: str | Coord | None = None,
    ) -> Coord:
        """Fix metadata of air_pressure coordinate (in-place).

        Parameters
        ----------
        cube:
            Input cube.
        coord:
            Coordinate for which metadata will be fixed in-place. If ``None``,
            assume the coordinate's name is `air_pressure`.

        Returns
        -------
        iris.coords.AuxCoord or iris.coords.DimCoord
            Fixed air_pressure coordinate. The coordinate is altered in-place;
            it is just returned out of convenience for easy access.

        """
        if coord is None:
            coord = cube.coord("air_pressure")
        elif isinstance(coord, str):
            coord = cube.coord(coord)
        coord.var_name = "plev"  # type: ignore
        coord.standard_name = "air_pressure"  # type: ignore
        coord.long_name = "pressure"  # type: ignore
        coord.convert_units("Pa")  # type: ignore
        coord.attributes["positive"] = "down"  # type: ignore
        return coord

    @staticmethod
    def fix_lat_metadata(
        cube: Cube,
        coord: str | Coord | None = None,
    ) -> Coord:
        """Fix metadata of latitude coordinate (in-place).

        Parameters
        ----------
        cube:
            Input cube.
        coord:
            Coordinate for which metadata will be fixed in-place. If ``None``,
            assume the coordinate's name is `latitude`.

        Returns
        -------
        iris.coords.AuxCoord or iris.coords.DimCoord
            Fixed latitude coordinate. The coordinate is altered in-place; it
            is just returned out of convenience for easy access.

        """
        if coord is None:
            coord = cube.coord("latitude")
        elif isinstance(coord, str):
            coord = cube.coord(coord)
        coord.var_name = "lat"  # type: ignore
        coord.standard_name = "latitude"  # type: ignore
        coord.long_name = "latitude"  # type: ignore
        coord.convert_units("degrees_north")  # type: ignore
        return coord

    @staticmethod
    def fix_lon_metadata(
        cube: Cube,
        coord: str | Coord | None = None,
    ) -> Coord:
        """Fix metadata of longitude coordinate (in-place).

        Parameters
        ----------
        cube:
            Input cube.
        coord:
            Coordinate for which metadata will be fixed in-place. If ``None``,
            assume the coordinate's name is `longitude`.

        Returns
        -------
        iris.coords.AuxCoord or iris.coords.DimCoord
            Fixed longitude coordinate. The coordinate is altered in-place; it
            is just returned out of convenience for easy access.

        """
        if coord is None:
            coord = cube.coord("longitude")
        elif isinstance(coord, str):
            coord = cube.coord(coord)
        coord.var_name = "lon"  # type: ignore
        coord.standard_name = "longitude"  # type: ignore
        coord.long_name = "longitude"  # type: ignore
        coord.convert_units("degrees_east")  # type: ignore
        return coord
