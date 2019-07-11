"""Calculation of sum of all black carbon (aerosol) tracers over all
   aerosol modes (ks, ki, as, cs)"""

from . import var_name_constraint


def derive(cubes):
    MP_BC_tot_cube = cubes.extract_strict(
        var_name_constraint('MP_BC_ki_ave')) + cubes.extract_strict(
            var_name_constraint('MP_BC_ks_ave')) + cubes.extract_strict(
                var_name_constraint('MP_BC_as_ave')) + cubes.extract_strict(
                    var_name_constraint('MP_BC_cs_ave'))
    return MP_BC_tot_cube
