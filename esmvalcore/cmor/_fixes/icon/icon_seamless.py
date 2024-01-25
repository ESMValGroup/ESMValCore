"""CMOR-like reformatting of ICON-Seamless (NWP physics)."""
import logging

from scipy import constants

from .icon import AllVars as IconAllVars
from ._base_fixes import IconFix, NegateData


logger = logging.getLogger(__name__)


AllVars = IconAllVars


# class Clwvi(IconFix):
#     """Fixes for ``clwvi``."""

#     def fix_metadata(self, cubes):
#         """Fix metadata."""
#         cube = (
#             self.get_cube(cubes, var_name='cllvi') +
#             self.get_cube(cubes, var_name='clivi')
#         )
#         cube.var_name = self.vardef.short_name
#         return CubeList([cube])


Hfls = NegateData


Hfss = NegateData


class Zg(IconFix):
    """Fixes for ``zg``."""

    def fix_metadata(self, cubes):
        """Fix metadata.

        Convert geopotential Phi given by ICON-Seamless to geopotential height
        Z using Z = Phi / g0 (g0 is standard acceleration of gravity).

        """
        g0_value = constants.value('standard acceleration of gravity')
        g0_units = constants.unit('standard acceleration of gravity')

        cube = self.get_cube(cubes)
        cube.data = cube.core_data() / g0_value
        cube.units /= g0_units

        return cubes
