"""Fixes for CLMcom-CCLM4-8-17 model."""
import iris
import numpy as np

from ..fix import Fix

import IPython
from traitlets.config import get_config
c = get_config()
c.InteractiveShellEmbed.colors = "Linux"


# class Sftlf(Fix):
#     """Fixes for tas."""

#     def fix_metadata(self, cubes):
#         """
#         Add missing height2m coordinate.

#         Parameters
#         ----------
#         cubes : iris.cube.CubeList
#             Input cubes.

#         Returns
#         -------
#         iris.cube.CubeList

#         """
#         IPython.embed(config=c)

#         for cube in cubes:
#             add_scalar_height_coord(cube)

#         return cubes


# class AllVars(Fix):
#     """Fixes for all variables."""
#     def fix_metadata(self, cubes):
#         """Fix metadata."""
#         fixed_cubes = iris.cube.CubeList()
#         for cube in cubes:
#             cube.data = cube.core_data().astype('float32')
#             fixed_cubes.append(cube)

#         return fixed_cubes
