"""Calculation of sum of all sulfate (aerosol) tracers over all
   aerosol modes (ns, ks, as, cs)"""

from . import var_name_constraint


def derive(cubes):
    MP_SO4mm_tot_cube = cubes.extract_strict(
        var_name_constraint('MP_SO4mm_ns_ave')) + cubes.extract_strict(
            var_name_constraint('MP_SO4mm_ks_ave')) + cubes.extract_strict(
                var_name_constraint('MP_SO4mm_as_ave')) + cubes.extract_strict(
                    var_name_constraint('MP_SO4mm_cs_ave'))
    return MP_SO4mm_tot_cube
