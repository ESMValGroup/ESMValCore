"""CMOR-like reformatting of ICON-XPP (NWP physics)."""

import logging

from iris.cube import CubeList
from scipy import constants

from ._base_fixes import IconFix, NegateData
from .icon import AllVars as IconAllVars

logger = logging.getLogger(__name__)


AllVars = IconAllVars


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


class Rsuscs(IconFix):
    """Fixes for ``rsuscs``."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        # Level at index 1 is surface
        cube = self.get_cube(cubes, var_name="swflx_up_clr")[:, 1, ...]
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
