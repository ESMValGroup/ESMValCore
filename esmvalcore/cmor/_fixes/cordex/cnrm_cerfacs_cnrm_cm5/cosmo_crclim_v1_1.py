from __future__ import annotations

from typing import TYPE_CHECKING

from esmvalcore.cmor.fix import Fix

if TYPE_CHECKING:
    from collections.abc import Sequence

    from iris.cube import Cube


class Snw(Fix):
    """Fixes for snw."""

    def fix_metadata(self, cubes: Sequence[Cube]) -> Sequence[Cube]:
        cube = self.get_cube_from_list(cubes)
        cube = cube.copy()
        cube.remove_coord("height")
        return [cube]
