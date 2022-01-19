"""Derivation of variable `rlntcs`."""
from iris import Constraint

from ._baseclass import DerivedVariableBase


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `rlntcs`."""

    @staticmethod
    def required(project):
        """Declare the variables needed for derivation."""
        required = [{'short_name': 'rlutcs'}]
        return required

    @staticmethod
    def calculate(cubes):
        """Compute toa net downward longwave radiation assuming clear sky."""
        rlutcs_cube = cubes.extract_cube(
            Constraint(name='toa_outgoing_longwave_flux_assuming_clear_sky'))
        rlutcs_cube.data = -rlutcs_cube.core_data()
        rlutcs_cube.attributes['positive'] = 'down'
        return rlutcs_cube
