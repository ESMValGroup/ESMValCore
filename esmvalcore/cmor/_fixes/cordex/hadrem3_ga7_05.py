"""Fixes for DMI-HIRHAM5 model."""
import iris

from ..common import ClFixHybridPressureCoord
from ..fix import Fix
from ..shared import add_scalar_height_coord

# import IPython
# from traitlets.config import get_config
# c = get_config()
# c.InteractiveShellEmbed.colors = "Linux"
#         IPython.embed(config=c)


from esmvalcore.preprocessor._shared import guess_bounds
import numpy as np


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
