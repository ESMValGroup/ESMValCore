"""Fixes for DMI-HIRHAM5 model."""
import iris

from ..common import ClFixHybridPressureCoord
from ..fix import Fix
from ..shared import add_scalar_height_coord

# import IPython
# from traitlets.config import get_config
# c = get_config()
# c.InteractiveShellEmbed.colors = "Linux"


class Tas(Fix):
    """Fixes for tas."""

    def fix_metadata(self, cubes):
        """
        Add missing height2m coordinate.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList

        """
        for cube in cubes:
            add_scalar_height_coord(cube)

        return cubes


# class AllVars(Fix):
#     """Fixes for all variables."""

#     def fix_metadata(self, cubes):
#         """Fix metadata."""
#         fixed_cubes = iris.cube.CubeList()

#         IPython.embed(config=c)
            #  No idea, but there is only one cube
#         for cube in cubes:
#             if len(cube.dim_coords) < 2 and len(cube.shape) > 1:
#                 print('here')
#                 IPython.embed(config=c)

#         return fixed_cubes

