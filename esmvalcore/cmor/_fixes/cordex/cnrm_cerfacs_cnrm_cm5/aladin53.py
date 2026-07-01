"""Fixes for rcm ALADIN53 driven by CNRM-CERFACS-CNRM-CM5."""

from __future__ import annotations

from typing import TYPE_CHECKING

import iris.coord_systems

from esmvalcore.cmor.fix import Fix

if TYPE_CHECKING:
    from collections.abc import Sequence

    from iris.cube import Cube


class Sftlf(Fix):
    """Fixes for sftlf."""

    def fix_metadata(self, cubes: Sequence[Cube]) -> Sequence[Cube]:
        for cube in cubes:
            cube.units = "1"
            cube.convert_units(self.vardef.units)
        return cubes


class Ts(Fix):
    """Fixes for ts."""

    def fix_metadata(self, cubes: Sequence[Cube]) -> Sequence[Cube]:
        for cube in cubes:
            cube.units = "deg_C"
            cube.convert_units(self.vardef.units)
        return cubes


class AllVars(Fix):
    """Fixes for all variables."""

    def fix_metadata(self, cubes: Sequence[Cube]) -> Sequence[Cube]:
        for cube in cubes:
            if self.extra_facets.get("domain") == "EUR-11":
                # Set false_easting and false_northing to 0.0, as odd values
                # have been observed in some files.
                coord_system = iris.coord_systems.LambertConformal(
                    central_lat=49.5,
                    central_lon=10.5,
                    secant_latitudes=(49.5,),
                )
                for coord_name in [
                    "projection_x_coordinate",
                    "projection_y_coordinate",
                ]:
                    if cube.coords(coord_name):
                        cube.coord(coord_name).coord_system = coord_system
        return cubes
