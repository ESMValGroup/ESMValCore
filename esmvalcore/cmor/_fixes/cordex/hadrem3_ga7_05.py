"""Fixes for MOHC HadREM3-GA7-05 model."""

import iris

from ..fix import Fix


class AllVars(Fix):
    """Fixes for all variables."""
    def fix_metadata(self, cubes):
        """Fix metadata."""

        fixed_cubes = iris.cube.CubeList()
        for cube in cubes:
            cube.coord('latitude').var_name = 'lat'
            cube.coord('longitude').var_name = 'lon'
            fixed_cubes.append(cube)

        return fixed_cubes
