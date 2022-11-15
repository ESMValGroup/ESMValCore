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
from shutil import copyfile

import dask.array as da
import iris.analysis
import iris.util
from iris import NameConstraint
from iris.aux_factory import HybridPressureFactory
from iris.cube import CubeList
from netCDF4 import Dataset
from scipy import constants

from ..shared import add_aux_coords_from_cubes
from ._base_fixes import EmacFix, NegateData, SetUnitsTo1

logger = logging.getLogger(__name__)


class AllVars(EmacFix):
    """Fixes for all variables."""

    # Dictionary to map invalid units in the data to valid entries
    INVALID_UNITS = {
        'kg/m**2s': 'kg m-2 s-1',
    }

    def fix_file(self, filepath, output_dir):
        """Fix file.

        Fixes hybrid pressure level coordinate.

        Note
        ----
        This fix removes the ``formula_terms`` attribute of the hybrid pressure
        level variables to make the corresponding coefficients appear correctly
        in the class:`iris.cube.CubeList` object returned by :mod:`iris.load`.

        """
        if 'alevel' not in self.vardef.dimensions:
            return filepath
        new_path = self.get_fixed_filepath(output_dir, filepath)
        copyfile(filepath, new_path)
        with Dataset(new_path, mode='a') as dataset:
            if 'formula_terms' in dataset.variables['lev'].ncattrs():
                del dataset.variables['lev'].formula_terms
            if 'formula_terms' in dataset.variables['ilev'].ncattrs():
                del dataset.variables['ilev'].formula_terms
        return new_path

    def fix_metadata(self, cubes):
        """Fix metadata."""
        cube = self.get_cube(cubes)

        # Fix time, latitude, and longitude coordinates
        self.fix_regular_time(cube)
        self.fix_regular_lat(cube)
        self.fix_regular_lon(cube)

        # Fix regular pressure levels (considers plev19, plev39, etc.)
        for dim_name in self.vardef.dimensions:
            if 'plev' in dim_name:
                self._fix_plev(cube)
                break

        # Fix hybrid pressure levels
        if 'alevel' in self.vardef.dimensions:
            cube = self._fix_alevel(cube, cubes)

        # Fix scalar coordinates
        self.fix_scalar_coords(cube)

        # Fix metadata of variable
        self.fix_var_metadata(cube)

        return CubeList([cube])

    def _fix_plev(self, cube):
        """Fix regular pressure level coordinate of cube."""
        for coord in cube.coords():
            coord_type = iris.util.guess_coord_axis(coord)

            if coord_type != 'Z':
                continue
            if not coord.units.is_convertible('Pa'):
                continue

            self.fix_plev_metadata(cube, coord)

            return

        raise ValueError(
            f"Cannot find requested pressure level coordinate for variable "
            f"'{self.vardef.short_name}', searched for Z-coordinates with "
            f"units that are convertible to Pa")

    @staticmethod
    def _fix_alevel(cube, cubes):
        """Fix hybrid pressure level coordinate of cube."""
        # Add coefficients for hybrid pressure level coordinate
        coords_to_add = {
            'hyam': 1,
            'hybm': 1,
            'aps_ave': (0, 2, 3),
        }
        add_aux_coords_from_cubes(cube, cubes, coords_to_add)

        # Reverse entire cube along Z-axis so that index 0 is surface level
        # Note: This would automatically be fixed by the CMOR checker, but this
        # fails to fix the bounds of ap and b
        cube = iris.util.reverse(cube, cube.coord(var_name='lev'))

        # Adapt metadata of coordinates
        lev_coord = cube.coord(var_name='lev')
        ap_coord = cube.coord(var_name='hyam')
        b_coord = cube.coord(var_name='hybm')
        ps_coord = cube.coord(var_name='aps_ave')

        lev_coord.var_name = 'lev'
        lev_coord.standard_name = 'atmosphere_hybrid_sigma_pressure_coordinate'
        lev_coord.long_name = 'hybrid sigma pressure coordinate'
        lev_coord.units = '1'
        lev_coord.attributes['positive'] = 'down'

        ap_coord.var_name = 'ap'
        ap_coord.standard_name = None
        ap_coord.long_name = 'vertical coordinate formula term: ap(k)'
        ap_coord.attributes = {}

        b_coord.var_name = 'b'
        b_coord.standard_name = None
        b_coord.long_name = 'vertical coordinate formula term: b(k)'
        b_coord.attributes = {}

        ps_coord.var_name = 'ps'
        ps_coord.standard_name = 'surface_air_pressure'
        ps_coord.long_name = 'Surface Air Pressure'
        ps_coord.attributes = {}

        # Add bounds for coefficients
        # (make sure to reverse cubes beforehand so index 0 is surface level)
        ap_bnds_cube = iris.util.reverse(
            cubes.extract_cube(NameConstraint(var_name='hyai')),
            0,
        )
        b_bnds_cube = iris.util.reverse(
            cubes.extract_cube(NameConstraint(var_name='hybi')),
            0,
        )
        ap_bounds = da.stack(
            [ap_bnds_cube.core_data()[:-1], ap_bnds_cube.core_data()[1:]],
            axis=-1,
        )
        b_bounds = da.stack(
            [b_bnds_cube.core_data()[:-1], b_bnds_cube.core_data()[1:]],
            axis=-1,
        )
        ap_coord.bounds = ap_bounds
        b_coord.bounds = b_bounds

        # Convert arrays to float64
        for coord in (ap_coord, b_coord, ps_coord):
            coord.points = coord.core_points().astype(
                float, casting='same_kind')
            if coord.bounds is not None:
                coord.bounds = coord.core_bounds().astype(
                    float, casting='same_kind')

        # Fix values of lev coordinate
        # Note: lev = a + b with a = ap / p0 (p0 = 100000 Pa)
        lev_coord.points = (ap_coord.core_points() / 100000.0 +
                            b_coord.core_points())
        lev_coord.bounds = (ap_coord.core_bounds() / 100000.0 +
                            b_coord.core_bounds())

        # Add HybridPressureFactory
        pressure_coord_factory = HybridPressureFactory(
            delta=ap_coord,
            sigma=b_coord,
            surface_air_pressure=ps_coord,
        )
        cube.add_aux_factory(pressure_coord_factory)

        return cube


Cl = SetUnitsTo1


Clt = SetUnitsTo1


class Clwvi(EmacFix):
    """Fixes for ``clwvi``."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        cube = (
            self.get_cube(cubes, var_name=['xlvi_cav', 'xlvi_ave']) +
            self.get_cube(cubes, var_name=['xivi_cav', 'xivi_ave'])
        )
        cube.var_name = self.vardef.short_name
        return CubeList([cube])


Evspsbl = NegateData


Hfls = NegateData


Hfss = NegateData


Hurs = SetUnitsTo1


class Od550aer(SetUnitsTo1):
    """Fixes for ``od550aer``."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        cubes = super().fix_metadata(cubes)
        cube = self.get_cube(cubes)
        z_coord = cube.coord(axis='Z')
        cube = cube.collapsed(z_coord, iris.analysis.SUM)
        return CubeList([cube])


class Pr(EmacFix):
    """Fixes for ``pr``."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        cube = (
            self.get_cube(cubes, var_name=['aprl_cav', 'aprl_ave']) +
            self.get_cube(cubes, var_name=['aprc_cav', 'aprc_ave'])
        )
        cube.var_name = self.vardef.short_name
        return CubeList([cube])


class Rlds(EmacFix):
    """Fixes for ``rlds``."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        cube = (
            self.get_cube(cubes, var_name=['flxtbot_cav', 'flxtbot_ave']) -
            self.get_cube(cubes, var_name=['tradsu_cav', 'tradsu_ave'])
        )
        cube.var_name = self.vardef.short_name
        return CubeList([cube])


Rlus = NegateData


Rlut = NegateData


Rlutcs = NegateData


class Rsds(EmacFix):
    """Fixes for ``rsds``."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        cube = (
            self.get_cube(cubes, var_name=['flxsbot_cav', 'flxsbot_ave']) -
            self.get_cube(cubes, var_name=['sradsu_cav', 'sradsu_ave'])
        )
        cube.var_name = self.vardef.short_name
        return CubeList([cube])


class Rsdt(EmacFix):
    """Fixes for ``rsdt``."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        cube = (
            self.get_cube(cubes, var_name=['flxstop_cav', 'flxstop_ave']) -
            self.get_cube(cubes, var_name=['srad0u_cav', 'srad0u_ave'])
        )
        cube.var_name = self.vardef.short_name
        return CubeList([cube])


Rsus = NegateData


Rsut = NegateData


Rsutcs = NegateData


class Rtmt(EmacFix):
    """Fixes for ``rtmt``."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        cube = (
            self.get_cube(cubes, var_name=['flxttop_cav', 'flxttop_ave']) +
            self.get_cube(cubes, var_name=['flxstop_cav', 'flxstop_ave'])
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


class Toz(EmacFix):
    """Fixes for ``tosga``."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        # Convert DU to mm
        # Note: 1 mm = 100 DU
        cube = self.get_cube(cubes)
        cube.data = cube.core_data() / 100.0
        cube.units = 'mm'
        return CubeList([cube])


class Zg(EmacFix):
    """Fixes for ``zg``."""

    def fix_metadata(self, cubes):
        """Fix metadata.

        Convert geopotential Phi given by EMAC to geopotential height Z using
        Z = Phi / g0 (g0 is standard acceleration of gravity)

        """
        g0_value = constants.value('standard acceleration of gravity')
        g0_units = constants.unit('standard acceleration of gravity')

        cube = self.get_cube(cubes)
        cube.data = cube.core_data() / g0_value
        cube.units /= g0_units

        return cubes


# Tracers


class MP_BC_tot(EmacFix):  # noqa: N801
    """Fixes for ``MP_BC_tot``."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        cube = (
            self.get_cube(cubes, var_name=['MP_BC_ki_cav', 'MP_BC_ki_ave']) +
            self.get_cube(cubes, var_name=['MP_BC_ks_cav', 'MP_BC_ks_ave']) +
            self.get_cube(cubes, var_name=['MP_BC_as_cav', 'MP_BC_as_ave']) +
            self.get_cube(cubes, var_name=['MP_BC_cs_cav', 'MP_BC_cs_ave'])
        )
        cube.var_name = self.vardef.short_name
        return CubeList([cube])


class MP_DU_tot(EmacFix):  # noqa: N801
    """Fixes for ``MP_DU_tot``."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        cube = (
            self.get_cube(cubes, var_name=['MP_DU_ai_cav', 'MP_DU_ai_ave']) +
            self.get_cube(cubes, var_name=['MP_DU_as_cav', 'MP_DU_as_ave']) +
            self.get_cube(cubes, var_name=['MP_DU_ci_cav', 'MP_DU_ci_ave']) +
            self.get_cube(cubes, var_name=['MP_DU_cs_cav', 'MP_DU_cs_ave'])
        )
        cube.var_name = self.vardef.short_name
        return CubeList([cube])


class MP_SO4mm_tot(EmacFix):  # noqa: N801
    """Fixes for ``MP_SO4mm_tot``."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        cube = (
            self.get_cube(
                cubes, var_name=['MP_SO4mm_ns_cav', 'MP_SO4mm_ns_ave']) +
            self.get_cube(
                cubes, var_name=['MP_SO4mm_ks_cav', 'MP_SO4mm_ks_ave']) +
            self.get_cube(
                cubes, var_name=['MP_SO4mm_as_cav', 'MP_SO4mm_as_ave']) +
            self.get_cube(
                cubes, var_name=['MP_SO4mm_cs_cav', 'MP_SO4mm_cs_ave'])
        )
        cube.var_name = self.vardef.short_name
        return CubeList([cube])


class MP_SS_tot(EmacFix):  # noqa: N801
    """Fixes for ``MP_SS_tot``."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        cube = (
            self.get_cube(cubes, var_name=['MP_SS_ks_cav', 'MP_SS_ks_ave']) +
            self.get_cube(cubes, var_name=['MP_SS_as_cav', 'MP_SS_as_ave']) +
            self.get_cube(cubes, var_name=['MP_SS_cs_cav', 'MP_SS_cs_ave'])
        )
        cube.var_name = self.vardef.short_name
        return CubeList([cube])
