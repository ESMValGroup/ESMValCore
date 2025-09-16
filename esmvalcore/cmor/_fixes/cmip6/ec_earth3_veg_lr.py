"""Fixes for EC-Earth3-Veg-LR model."""

from collections.abc import Sequence

import iris.cube

from esmvalcore.cmor._fixes.common import OceanFixGrid
from esmvalcore.cmor._fixes.fix import Fix


class AllVars(Fix):
    """Fixes for all variables."""

    def fix_metadata(
        self,
        cubes: Sequence[iris.cube.Cube],
    ) -> Sequence[iris.cube.Cube]:
        """Use the same long name for horizontal index coordinates."""
        for cube in cubes:
            # Use the same long name for index coordinates in all files to avoid
            # problems when concatenating.
            for var_name, dim_name in [("i", "first"), ("j", "second")]:
                if coords := cube.coords(var_name=var_name, dim_coords=True):
                    long_name = f"cell index along {dim_name} dimension"
                    coords[0].long_name = long_name
        return cubes


Siconc = OceanFixGrid
