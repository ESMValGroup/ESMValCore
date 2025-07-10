"""CMOR-like reformatting of ICON-A (ECHAM physics)."""

import logging

from iris.cube import CubeList

from ._base_fixes import AllVarsBase, IconFix, NegateData

logger = logging.getLogger(__name__)


AllVars = AllVarsBase


class Clwvi(IconFix):
    """Fixes for ``clwvi``."""

    def fix_metadata(self, cubes: CubeList) -> CubeList:
        """Fix metadata."""
        cube = self.get_cube(cubes, var_name="cllvi") + self.get_cube(
            cubes,
            var_name="clivi",
        )
        cube.var_name = self.vardef.short_name
        return CubeList([cube])


Hfls = NegateData


Hfss = NegateData


class Rtmt(IconFix):
    """Fixes for ``rtmt``."""

    def fix_metadata(self, cubes: CubeList) -> CubeList:
        """Fix metadata."""
        cube = (
            self.get_cube(cubes, var_name="rsdt")
            - self.get_cube(cubes, var_name="rsut")
            - self.get_cube(cubes, var_name="rlut")
        )
        cube.var_name = self.vardef.short_name
        return CubeList([cube])


Rtnt = Rtmt
