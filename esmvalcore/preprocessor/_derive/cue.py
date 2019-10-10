"""Derivation of variable `cue`."""

import iris
from ._baseclass import DerivedVariableBase


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `cue` (Carbon Use Efficiency)."""

    # Required variables
    required = [{'short_name': 'gpp', 'short_name': 'npp',}]
    # npp = net primary production
    # gpp = gross primary production

    @staticmethod
    def calculate(cubes):
        """Compute Carbon Use Efficiency.

        """
        gpp_cube = cubes.extract_strict(
            iris.Constraint(short_name='gpp'))
        et_cube = cubes.extract_strict(
            iris.Constraint(short_name='et'))

        return et_cube/gpp_cube
