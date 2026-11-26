"""Fixes for rcm HadREM3-GA7-05 driven by CNRM-CERFACS-CNRM-CM5."""

from __future__ import annotations

from typing import TYPE_CHECKING

import iris
import iris.cube

from esmvalcore.cmor._fixes.cordex.cordex_fixes import (
    MOHCHadREM3GA705 as BaseFix,
)
from esmvalcore.cmor.fix import Fix

if TYPE_CHECKING:
    from collections.abc import Sequence

AllVars = BaseFix


class Sic(Fix):
    """Fixes for variable sic."""

    def fix_metadata(
        self,
        cubes: Sequence[iris.cube.Cube],
    ) -> Sequence[iris.cube.Cube]:
        cube = self.get_cube_from_list(iris.cube.CubeList(cubes)).copy()
        cube.units = 1
        cube.convert_units("%")
        return [cube]
