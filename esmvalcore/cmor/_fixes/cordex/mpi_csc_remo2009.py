"""Fixes for CLMcom-CCLM4-8-17 model."""
import iris
import numpy as np

from ..fix import Fix

class AllVars(Fix):
    """Fixes for all variables."""
    def fix_metadata(self, cubes):
        """Fix metadata."""
        fixed_cubes = iris.cube.CubeList()
        for cube in cubes:
            cube.data = cube.core_data().astype('float32')
            fixed_cubes.append(cube)

        return fixed_cubes
