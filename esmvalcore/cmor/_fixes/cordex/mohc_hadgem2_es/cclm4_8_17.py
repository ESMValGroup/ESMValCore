"""Fixes for rcm CCLM4-8-17 driven by MOHC-HadGEM2-ES."""

from __future__ import annotations

from typing import TYPE_CHECKING

import iris
import iris.cube

from esmvalcore.cmor._fixes.cordex.cordex_fixes import (
    CLMcomCCLM4817 as BaseFix,
)
from esmvalcore.cmor.fix import Fix

if TYPE_CHECKING:
    from collections.abc import Sequence

AllVars = BaseFix


class Clivi(Fix):
    """Fixes for variable clivi."""

    def fix_metadata(
        self,
        cubes: Sequence[iris.cube.Cube],
    ) -> Sequence[iris.cube.Cube]:
        cube = self.get_cube_from_list(iris.cube.CubeList(cubes)).copy()
        cube.units = "Mg m-2"
        cube.convert_units("kg m-2")
        return [cube]
