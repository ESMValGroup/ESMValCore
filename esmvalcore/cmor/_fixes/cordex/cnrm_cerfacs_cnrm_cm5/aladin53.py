"""Fixes for rcm ALADIN53 driven by CNRM-CERFACS-CNRM-CM5."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from iris.util import promote_aux_coord_to_dim_coord

from esmvalcore.cmor.fix import Fix

if TYPE_CHECKING:
    from collections.abc import Sequence

    from iris.cube import Cube


class AllVars(Fix):
    """Fixes for all variables."""

    def fix_metadata(self, cubes: Sequence[Cube]) -> Sequence[Cube]:
        domain_step = {
            "11": 12500,
            "22": 25000,
            "44": 50000,
        }

        for cube in cubes:
            domain_resolution = self.extra_facets["domain"].split("-")[-1]
            step = domain_step[domain_resolution]

            for coord_name in [
                "projection_x_coordinate",
                "projection_y_coordinate",
            ]:
                coord = cube.coord(coord_name)
                n_steps = coord.shape[0]
                coord.points = step * np.linspace(
                    -(n_steps - 1) / 2,
                    (n_steps - 1) / 2,
                    n_steps,
                )
                coord.units = "m"
                coord.guess_bounds()
                promote_aux_coord_to_dim_coord(cube, coord_name)

        return cubes


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
