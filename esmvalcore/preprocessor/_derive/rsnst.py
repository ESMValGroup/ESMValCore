"""Derivation of variable `rsnst`.

authors:
    - weig_ka

"""
from iris import Constraint

from ._baseclass import DerivedVariableBase


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `rsnst`."""

    @staticmethod
    def required(project):
        """Declare the variables needed for derivation."""
        required = [
            {
                'short_name': 'rsds'
            },
            {
                'short_name': 'rsdt'
            },
            {
                'short_name': 'rsus'
            },
            {
                'short_name': 'rsut'
            },
        ]
        return required

    @staticmethod
    def calculate(cubes):
        """Compute Heating from Shortwave Absorption."""
        rsds_cube = cubes.extract_strict(
            Constraint(name='surface_downwelling_shortwave_flux_in_air'))
        rsdt_cube = cubes.extract_strict(
            Constraint(name='toa_incoming_shortwave_flux'))
        rsus_cube = cubes.extract_strict(
            Constraint(name='surface_upwelling_shortwave_flux_in_air'))
        rsut_cube = cubes.extract_strict(
            Constraint(name='toa_outgoing_shortwave_flux'))

        rsnst_cube = (rsdt_cube - rsut_cube) - (rsds_cube - rsus_cube)

        return rsnst_cube
