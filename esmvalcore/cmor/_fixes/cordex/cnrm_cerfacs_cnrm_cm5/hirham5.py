"""Fixes for rcm HIRHAM5 driven by CNRM-CERFACS-CNRM-CM5."""

from __future__ import annotations

from typing import TYPE_CHECKING

import iris
import iris.cube

from esmvalcore.cmor.fix import Fix

if TYPE_CHECKING:
    from collections.abc import Sequence


class Clivi(Fix):
    """Fixes for variable clivi."""

    def fix_metadata(
        self,
        cubes: Sequence[iris.cube.Cube],
    ) -> Sequence[iris.cube.Cube]:
        cube = self.get_cube_from_list(iris.cube.CubeList(cubes)).copy()
        cube.units = "g m-2"
        cube.convert_units("kg m-2")
        return [cube]
