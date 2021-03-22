"""Derivation of variable `rsus`.

authors:
    - lukas_brunner

"""
from iris import Constraint

from ._baseclass import DerivedVariableBase


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `rsus`."""

    @staticmethod
    def required(project):
        """Declare the variables needed for derivation."""
        required = [
            {
                'short_name': 'rsds'
            },
            {
                'short_name': 'rsns'
            },
        ]
        return required

    @staticmethod
    def calculate(cubes):
        """Compute upwelling shortwave flux from downwelling and net."""
        rsds_cube = cubes.extract_cube(
            Constraint(name='surface_downwelling_shortwave_flux_in_air'))
        rsns_cube = cubes.extract_cube(
            Constraint(name='surface_net_downward_shortwave_flux'))
        # fix latitude and longitude var_name
        rsns_cube.coord("longitude").var_name = "Longitude"
        rsns_cube.coord("latitude").var_name = "Latitude"

        rsus_cube = rsds_cube - rsns_cube

        return rsus_cube
