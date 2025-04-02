"""CMOR-like reformatting of ICON-XPP (NWP physics)."""

import logging

from iris.cube import CubeList
from scipy import constants

from ..shared import fix_ocean_depth_coord
from ._base_fixes import AllVarsBase, IconFix, NegateData

logger = logging.getLogger(__name__)


class AllVars(AllVarsBase):
    """Fixes necessary for all ICON-XPP variables."""

    DEFAULT_PFULL_VAR_NAME = "pres"


class Clwvi(IconFix):
    """Fixes for ``clwvi``."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        cube = self.get_cube(cubes, var_name="tqc_dia") + self.get_cube(
            cubes, var_name="tqi_dia"
        )
        cube.var_name = self.vardef.short_name
        return CubeList([cube])


Evspsbl = NegateData


Hfls = NegateData


Hfss = NegateData


Rlut = NegateData


class Rlutcs(IconFix):
    """Fixes for ``rlutcs``."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        # Level at index 0 is TOA
        cube = self.get_cube(cubes, var_name="lwflx_up_clr")[:, 0, ...]
        cube.remove_coord("height")
        return CubeList([cube])


class Rsutcs(IconFix):
    """Fixes for ``rsutcs``."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        # Level at index 0 is TOA
        cube = self.get_cube(cubes, var_name="swflx_up_clr")[:, 0, ...]
        cube.remove_coord("height")
        return CubeList([cube])


class Rtmt(IconFix):
    """Fixes for ``rtmt``."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        cube = self.get_cube(cubes, var_name="sob_t") + self.get_cube(
            cubes, var_name="thb_t"
        )
        cube.var_name = self.vardef.short_name
        return CubeList([cube])


Rtnt = Rtmt


class Zg(IconFix):
    """Fixes for ``zg``."""

    def fix_metadata(self, cubes):
        """Fix metadata.

        Convert geopotential Phi given by ICON-XPP to geopotential height
        Z using Z = Phi / g0 (g0 is standard acceleration of gravity).

        """
        g0_value = constants.value("standard acceleration of gravity")
        g0_units = constants.unit("standard acceleration of gravity")

        cube = self.get_cube(cubes)
        cube.data = cube.core_data() / g0_value
        cube.units /= g0_units

        return cubes


class Omon(IconFix):
    """Fixes for ocean variables."""

    def fix_metadata(self, cubes):
        """Fix ocean depth coordinate.

        Parameters
        ----------
        cubes: iris CubeList
            List of cubes to fix

        Returns
        -------
        iris.cube.CubeList

        """
        for cube in cubes:
            if cube.coords(axis="Z"):
                z_coord = cube.coord(axis="Z")
                if z_coord.var_name == "depth":
                    fix_ocean_depth_coord(cube)
        return cubes


Oyr = Omon
Oday = Omon
