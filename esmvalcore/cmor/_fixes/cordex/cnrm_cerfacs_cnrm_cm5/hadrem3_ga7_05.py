"""Fixes for rcm HadREM3-GA7-05 driven by CNRM-CERFACS-CNRM-CM5."""

from __future__ import annotations

from typing import TYPE_CHECKING

from esmvalcore.cmor._fixes.cordex.cordex_fixes import (
    MOHCHadREM3GA705 as BaseFix,
)
from esmvalcore.cmor._fixes.fix import Fix

if TYPE_CHECKING:
    from collections.abc import Sequence

    from iris.cube import Cube


AllVars = BaseFix


class Sftlf(Fix):
    """Fixes for sftlf."""

    def fix_metadata(self, cubes: Sequence[Cube]) -> Sequence[Cube]:
        """Fix metadata."""
        cube = self.get_cube_from_list(cubes, "sftlf")
        cube = cube.copy()
        cube.long_name = self.vardef.long_name
        return [cube]
