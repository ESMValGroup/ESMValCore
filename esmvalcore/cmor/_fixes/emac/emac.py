"""On-the-fly CMORizer for EMAC.

Note
----
For many variables, derivations from multiple variables (i.e., an output
variable is calculated from multiple other variables) are necessary. These are
implemented in ``fix_metadata``, not in ``fix_data``, here. The reason for this
is that ``fix_metadata`` takes all cubes (and thus all input variables of the
input file) as argument while ``fix_data`` only takes one cube (the output
variable) as single argument.

"""

import logging

import dask.array as da
import iris.analysis
import iris.util
from iris import NameConstraint
from iris.cube import CubeList

from ..shared import (
    add_scalar_height_coord,
    add_scalar_lambda550nm_coord,
    add_scalar_typesi_coord,
)
from ._base_fixes import EmacFix, NegateData, SetUnitsTo1, SetUnitsTo1SumOverZ

logger = logging.getLogger(__name__)


INVALID_UNITS = {
    'kg/m**2s': 'kg m-2 s-1',
}


class AllVars(EmacFix):
    """Fixes for all variables."""

    def fix_data(self, cube):
        """Fix data."""
        # Fix mask by masking all values where the absolute value is greater
        # than a given threshold (affects mostly 3D variables)
        mask_threshold = 1e20
        cube.data = da.ma.masked_outside(
            cube.core_data(), -mask_threshold, mask_threshold,
        )
        return cube

    def fix_metadata(self, cubes):
        """Fix metadata."""
        cube = self.get_cube(cubes)

        # Fix time
        if 'time' in self.vardef.dimensions:
            self._fix_time(cube)

        # Fix pressure levels (considers plev19, plev39, etc.)
        for dim_name in self.vardef.dimensions:
            if 'plev' in dim_name:
                self._fix_plev(cube)
                break

        # Fix latitude
        if 'latitude' in self.vardef.dimensions:
            self._fix_lat(cube)

        # Fix longitude
        if 'longitude' in self.vardef.dimensions:
            self._fix_lon(cube)

        # Fix scalar coordinates
        self._fix_scalar_coords(cube)

        # Fix metadata of variable
        self._fix_var_metadata(cube)

        return CubeList([cube])

    @staticmethod
    def _fix_lat(cube):
        """Fix latitude coordinate of cube."""
        lat = cube.coord('latitude')
        lat.var_name = 'lat'
        lat.standard_name = 'latitude'
        lat.long_name = 'latitude'
        lat.convert_units('degrees_north')

        # Add bounds if possible (not possible if cube only contains single
        # lat point)
        if not lat.has_bounds():
            try:
                lat.guess_bounds()
            except ValueError:
                pass

    @staticmethod
    def _fix_lon(cube):
        """Fix longitude coordinate of cube."""
        lon = cube.coord('longitude')
        lon.var_name = 'lon'
        lon.standard_name = 'longitude'
        lon.long_name = 'longitude'
        lon.convert_units('degrees_east')

        # Add bounds if possible (not possible if cube only contains single
        # lon point)
        if not lon.has_bounds():
            try:
                lon.guess_bounds()
            except ValueError:
                pass

    def _fix_plev(self, cube):
        """Fix pressure level coordinate of cube."""
        for coord in cube.coords():
            coord_type = iris.util.guess_coord_axis(coord)
            if coord_type != 'Z':
                continue
            if not coord.units.is_convertible('Pa'):
                continue
            coord.var_name = 'plev'
            coord.standard_name = 'air_pressure'
            coord.long_name = 'pressure'
            coord.convert_units('Pa')
            coord.attributes['positive'] = 'down'
            return
        raise ValueError(
            f"Cannot find requested pressure level coordinate for variable "
            f"'{self.vardef.short_name}', searched for Z-coordinates with "
            f"units that are convertible to Pa")

    def _fix_scalar_coords(self, cube):
        """Fix scalar coordinates."""
        if 'height2m' in self.vardef.dimensions:
            add_scalar_height_coord(cube, 2.0)
        if 'height10m' in self.vardef.dimensions:
            add_scalar_height_coord(cube, 10.0)
        if 'lambda550nm' in self.vardef.dimensions:
            add_scalar_lambda550nm_coord(cube)
        if 'typesi' in self.vardef.dimensions:
            add_scalar_typesi_coord(cube, 'sea_ice')

    @staticmethod
    def _fix_time(cube):
        """Fix time coordinate of cube."""
        time_coord = cube.coord('time')
        time_coord.var_name = 'time'
        time_coord.standard_name = 'time'
        time_coord.long_name = 'time'

        # Add bounds if possible (not possible if cube only contains single
        # time point)
        if not time_coord.has_bounds():
            try:
                time_coord.guess_bounds()
            except ValueError:
                pass

    def _fix_var_metadata(self, cube):
        """Fix metadata of variable."""
        if self.vardef.standard_name == '':
            cube.standard_name = None
        else:
            cube.standard_name = self.vardef.standard_name
        cube.var_name = self.vardef.short_name
        cube.long_name = self.vardef.long_name

        # Fix units
        if 'invalid_units' in cube.attributes:
            invalid_units = cube.attributes.pop('invalid_units')
            new_units = INVALID_UNITS.get(
                invalid_units,
                invalid_units.replace('**', '^'),
            )
            try:
                cube.units = new_units
            except ValueError as exc:
                raise ValueError(
                    f"Failed to fix invalid units '{invalid_units}' for "
                    f"variable '{self.vardef.short_name}'") from exc
        if cube.units != self.vardef.units:
            cube.convert_units(self.vardef.units)


Clt = SetUnitsTo1


class Clwvi(EmacFix):
    """Fixes for ``clwvi``."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        cube = (
            cubes.extract_cube(NameConstraint(var_name='xlvi_ave')) +
            cubes.extract_cube(NameConstraint(var_name='xivi_ave'))
        )
        cube.var_name = self.vardef.short_name
        return CubeList([cube])


Evspsbl = NegateData


class MP_BC_tot(EmacFix):  # noqa: N801
    """Fixes for ``MP_BC_tot``."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        cube = (
            cubes.extract_cube(NameConstraint(var_name='MP_BC_ki_ave')) +
            cubes.extract_cube(NameConstraint(var_name='MP_BC_ks_ave')) +
            cubes.extract_cube(NameConstraint(var_name='MP_BC_as_ave')) +
            cubes.extract_cube(NameConstraint(var_name='MP_BC_cs_ave'))
        )
        cube.var_name = self.vardef.short_name
        return CubeList([cube])


class MP_DU_tot(EmacFix):  # noqa: N801
    """Fixes for ``MP_DU_tot``."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        cube = (
            cubes.extract_cube(NameConstraint(var_name='MP_DU_ai_ave')) +
            cubes.extract_cube(NameConstraint(var_name='MP_DU_as_ave')) +
            cubes.extract_cube(NameConstraint(var_name='MP_DU_ci_ave')) +
            cubes.extract_cube(NameConstraint(var_name='MP_DU_cs_ave'))
        )
        cube.var_name = self.vardef.short_name
        return CubeList([cube])


class MP_SO4mm_tot(EmacFix):  # noqa: N801
    """Fixes for ``MP_SO4mm_tot``."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        cube = (
            cubes.extract_cube(NameConstraint(var_name='MP_SO4mm_ns_ave')) +
            cubes.extract_cube(NameConstraint(var_name='MP_SO4mm_ks_ave')) +
            cubes.extract_cube(NameConstraint(var_name='MP_SO4mm_as_ave')) +
            cubes.extract_cube(NameConstraint(var_name='MP_SO4mm_cs_ave'))
        )
        cube.var_name = self.vardef.short_name
        return CubeList([cube])


class MP_SS_tot(EmacFix):  # noqa: N801
    """Fixes for ``MP_SS_tot``."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        cube = (
            cubes.extract_cube(NameConstraint(var_name='MP_SS_ks_ave')) +
            cubes.extract_cube(NameConstraint(var_name='MP_SS_as_ave')) +
            cubes.extract_cube(NameConstraint(var_name='MP_SS_cs_ave'))
        )
        cube.var_name = self.vardef.short_name
        return CubeList([cube])


Od550aer = SetUnitsTo1SumOverZ


class Pr(EmacFix):
    """Fixes for ``pr``."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        cube = (
            cubes.extract_cube(NameConstraint(var_name='aprl_ave')) +
            cubes.extract_cube(NameConstraint(var_name='aprc_ave')) +
            cubes.extract_cube(NameConstraint(var_name='aprs_ave'))
        )
        cube.var_name = self.vardef.short_name
        return CubeList([cube])


class Rlds(EmacFix):
    """Fixes for ``rlds``."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        cube = (
            cubes.extract_cube(NameConstraint(var_name='flxtbot_ave')) -
            cubes.extract_cube(NameConstraint(var_name='tradsu_ave'))
        )
        cube.var_name = self.vardef.short_name
        return CubeList([cube])


Rlus = NegateData


Rlut = NegateData


class Rsds(EmacFix):
    """Fixes for ``rsds``."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        cube = (
            cubes.extract_cube(NameConstraint(var_name='flxsbot_ave')) -
            cubes.extract_cube(NameConstraint(var_name='sradsu_ave'))
        )
        cube.var_name = self.vardef.short_name
        return CubeList([cube])


class Rsdt(EmacFix):
    """Fixes for ``rsdt``."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        cube = (
            cubes.extract_cube(NameConstraint(var_name='flxstop_ave')) -
            cubes.extract_cube(NameConstraint(var_name='srad0u_ave'))
        )
        cube.var_name = self.vardef.short_name
        return CubeList([cube])


Rsus = NegateData


Rsut = NegateData


class Rtmt(EmacFix):
    """Fixes for ``rtmt``."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        cube = (
            cubes.extract_cube(NameConstraint(var_name='flxttop_ave')) +
            cubes.extract_cube(NameConstraint(var_name='flxstop_ave'))
        )
        cube.var_name = self.vardef.short_name
        return CubeList([cube])


Siconc = SetUnitsTo1


Siconca = SetUnitsTo1


class Sithick(EmacFix):
    """Fixes for ``sithick``."""

    def fix_data(self, cube):
        """Fix data."""
        cube.data = da.ma.masked_equal(cube.core_data(), 0.0)
        return cube


class Tosga(EmacFix):
    """Fixes for ``tosga``."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        cube = self.get_cube(cubes)
        cube.units = 'celsius'
        return CubeList([cube])


class Toz(EmacFix):
    """Fixes for ``tosga``."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        # Note: 1 mm = 100 DU
        cube = self.get_cube(cubes)
        cube.data = cube.core_data() / 100.0
        cube.units = 'mm'
        return CubeList([cube])
