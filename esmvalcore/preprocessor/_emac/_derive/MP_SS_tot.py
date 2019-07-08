"""Calculation of sum of all sea salt (aerosol) tracers over all
   aerosol modes (ks, as, cs)"""

from . import var_name_constraint


def derive(cubes):
    MP_SS_tot_cube = cubes.extract_strict(
        var_name_constraint('MP_SS_ks')) + cubes.extract_strict(
            var_name_constraint('MP_SS_as')) + cubes.extract_strict(
                var_name_constraint('MP_SS_cs'))
    return MP_SS_tot_cube
