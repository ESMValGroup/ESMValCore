"""Calculation of sum of all sea salt (aerosol) tracers over all aerosol modes.

These are (ks, as, cs).

"""

from . import var_name_constraint


def derive(cubes):
    """Derive `MP_SS_tot`."""
    mp_ss_tot_cube = (
        cubes.extract_strict(var_name_constraint('MP_SS_ks_ave')) +
        cubes.extract_strict(var_name_constraint('MP_SS_as_ave')) +
        cubes.extract_strict(var_name_constraint('MP_SS_cs_ave')))
    return mp_ss_tot_cube
