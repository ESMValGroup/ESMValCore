"""CMOR-like reformatting of ICON-XPP (NWP physics)."""

import logging

from iris.cube import CubeList
from scipy import constants

from ._base_fixes import (
    AllVarsBase,
    BasinToVariableMapping,
    IconFix,
    NegateData,
    OceanBasinVariable,
)

logger = logging.getLogger(__name__)


class AllVars(AllVarsBase):
    """Fixes necessary for all ICON-XPP variables."""

    DEFAULT_PFULL_VAR_NAME = "pres"


class Clwvi(IconFix):
    """Fixes for ``clwvi``."""

    def fix_metadata(self, cubes: CubeList) -> CubeList:
        """Fix metadata."""
        cube = self.get_cube(cubes, var_name="tqc_dia") + self.get_cube(
            cubes,
            var_name="tqi_dia",
        )
        cube.var_name = self.vardef.short_name
        return CubeList([cube])


Evspsbl = NegateData


class Gpp(IconFix):
    """Fixes for ``gpp``."""

    def fix_metadata(self, cubes: CubeList) -> CubeList:
        """Fix metadata.

        Convert photosynthesis flux from `mol m-2 s-1` to `kgC m-2 s-1`.
        Molar mass of C: M(C) = 12.011 g/mol.

        """
        cube = self.get_cube(cubes)
        conversion_factor = 12.011 / 1000.0  # = M(C) / 1000 g/kg  [kgC/mol]
        cube.data = cube.core_data() * conversion_factor
        cube.units = "kg m-2 s-1"  # kgC m-2 s-1
        cube.attributes.pop("invalid_units", None)

        return cubes


class Hfbasin(OceanBasinVariable):
    """Fixes for ``hfbasin``."""

    basin_to_var = BasinToVariableMapping(
        atlantic_arctic_ocean="atlantic_hfbasin",
        indian_pacific_ocean="pacific_hfbasin",
        global_ocean="global_hfbasin",
    )


Hfls = NegateData


Hfss = NegateData


class Msftmz(OceanBasinVariable):
    """Fixes for ``msftmz``."""

    basin_to_var = BasinToVariableMapping(
        atlantic_arctic_ocean="atlantic_moc",
        indian_pacific_ocean="pacific_moc",
        global_ocean="global_moc",
    )


Rlut = NegateData


class Rlutcs(IconFix):
    """Fixes for ``rlutcs``."""

    def fix_metadata(self, cubes: CubeList) -> CubeList:
        """Fix metadata."""
        # Level at index 0 is TOA
        cube = self.get_cube(cubes, var_name="lwflx_up_clr")[:, 0, ...]
        cube.remove_coord("height")
        return CubeList([cube])


class Rsutcs(IconFix):
    """Fixes for ``rsutcs``."""

    def fix_metadata(self, cubes: CubeList) -> CubeList:
        """Fix metadata."""
        # Level at index 0 is TOA
        cube = self.get_cube(cubes, var_name="swflx_up_clr")[:, 0, ...]
        cube.remove_coord("height")
        return CubeList([cube])


class Rtmt(IconFix):
    """Fixes for ``rtmt``."""

    def fix_metadata(self, cubes: CubeList) -> CubeList:
        """Fix metadata."""
        cube = self.get_cube(cubes, var_name="sob_t") + self.get_cube(
            cubes,
            var_name="thb_t",
        )
        cube.var_name = self.vardef.short_name
        return CubeList([cube])


Rtnt = Rtmt


class Zg(IconFix):
    """Fixes for ``zg``."""

    def fix_metadata(self, cubes: CubeList) -> CubeList:
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
