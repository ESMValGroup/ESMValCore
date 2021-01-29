"""Derivation of variable `rsnstcs`.

authors:
    - weig_ka

"""
from cf_units import Unit
from iris import Constraint

from ._baseclass import DerivedVariableBase


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `rsnstcsnorm`."""

    @staticmethod
    def required(project):
        """Declare the variables needed for derivation."""
        required = [
            {
                'short_name': 'rsdscs'
            },
            {
                'short_name': 'rsdt'
            },
            {
                'short_name': 'rsuscs'
            },
            {
                'short_name': 'rsutcs'
            },
        ]
        return required

    @staticmethod
    def calculate(cubes):
        """
        Compute `rsnstcs`.

        Compute Heating from Shortwave Absorption
        assuming clear sky normalized by
        the incoming shortwave flux at the top of the atmosphere.
        """
        rsdscs_cube = cubes.extract_cube(
            Constraint(name='surface_downwelling_shortwave_flux_in_air_' +
                       'assuming_clear_sky'))
        rsdt_cube = cubes.extract_cube(
            Constraint(name='toa_incoming_shortwave_flux'))
        rsuscs_cube = cubes.extract_cube(
            Constraint(name='surface_upwelling_shortwave_flux_in_air_' +
                       'assuming_clear_sky'))
        rsutcs_cube = cubes.extract_cube(
            Constraint(name='toa_outgoing_shortwave_flux_assuming_clear_sky'))

        rsnstcsnorm_cube = (((rsdt_cube - rsutcs_cube) -
                             (rsdscs_cube - rsuscs_cube)) / rsdt_cube) * 100.0
        rsnstcsnorm_cube.units = Unit('percent')
        rsnstcsnorm_cube.attributes.pop('positive', None)

        return rsnstcsnorm_cube
