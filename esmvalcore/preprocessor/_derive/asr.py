"""Derivation of variable `asr`."""

from iris import Constraint

from ._baseclass import DerivedVariableBase


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `asr`."""

    # Required variables
    required = [{'short_name': 'rsdt'}, {'short_name': 'rsut'}]

    @staticmethod
    def calculate(cubes):
        """Compute absorbed shortwave radiation."""
        rsdt_cube = cubes.extract_strict(
            Constraint(name='toa_incoming_shortwave_flux'))
        rsut_cube = cubes.extract_strict(
            Constraint(name='toa_outgoing_shortwave_flux'))

        asr_cube = rsdt_cube - rsut_cube
        asr_cube.attributes['positive'] = 'down'

        return asr_cube
