"""Fixes for rcm ALADIN53 driven by CNRM-CERFACS-CNRM-CM5."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
