"""Derivation of variable `rlds`.

The variable 'rlds' (Surface Downwelling Longwave Radiation) is
a combination of the two EMAC variables: 'flxtbot_ave' and 'tradsu_ave'.
(following the recipe from the DKRZ CMIP6 Data Request WebGUI)
(https://c6dreq.dkrz.de/)
"""

from . import var_name_constraint


def derive(cubes):
    rlds_cube = cubes.extract_strict(
        var_name_constraint('flxtbot_ave')) - cubes.extract_strict(
            var_name_constraint('tradsu_ave'))

    return rlds_cube
