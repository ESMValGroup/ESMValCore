"""CMOR-like reformatting of ICON-XPP (NWP physics)."""

import logging

from iris.coords import AuxCoord
from iris.cube import CubeList
from scipy import constants

from ._base_fixes import AllVarsBase, IconFix, NegateData

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

        Convert photosynthesis flux from mol(co2) m-2 s-1 to kg m-2 s-1.
        Molar mass of CO2(kg) is 44.0095 g/mol

        """
        cube = self.get_cube(cubes)
        cube.data = cube.core_data() * 44.0095 / 1000
        cube.units = "kg m-2 s-1"
        cube.attributes.pop("invalid_units", None)

        return cubes


Hfls = NegateData


Hfss = NegateData


class Msftmz(IconFix):
    """Fixes for ``msftmz``."""

    def fix_metadata(self, cubes: CubeList) -> CubeList:
        """Fix metadata."""
        preprocessed_cubes = CubeList([])
        basin_coord = AuxCoord(
            "placeholder",
            standard_name="region",
            long_name="ocean basin",
            var_name="basin",
        )
        var_names = {
            "atlantic_moc": "atlantic_arctic_ocean",
            "pacific_moc": "indian_pacific_ocean",
            "global_moc": "global_ocean",
        }
        for var_name, basin in var_names.items():
            cube = self.get_cube(cubes, var_name=var_name)
            cube.var_name = "msftmz"
            cube.long_name = None
            cube.attributes.locals = {}

            # Remove longitude coordinate (with length 1)
            cube = cube[..., 0]
            cube.remove_coord("longitude")

            # Add scalar basin coordinate
            cube.add_aux_coord(basin_coord.copy(basin), ())
            preprocessed_cubes.append(cube)

        msftmz_cube = preprocessed_cubes.merge_cube()

        # Swap time and basin coordinates
        msftmz_cube.transpose([1, 0, 2, 3])

        # By default, merge_cube() sorts the coordinate alphabetically (i.e.,
        # atlantic_arctic_ocean -> global_ocean -> indian_pacific_ocean). Thus,
        # we need to restore the desired order (atlantic_arctic_ocean ->
        # indian_pacific_ocean -> global_ocean).
        msftmz_cube = msftmz_cube[:, [0, 2, 1], ...]

        return CubeList([msftmz_cube])


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
