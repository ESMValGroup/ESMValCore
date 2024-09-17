"""Derivation of variable `rlus`.

authors:
    - lukas_brunner

"""
from iris import Constraint

from ._baseclass import DerivedVariableBase


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `rlus`."""

    @staticmethod
    def required(project):
        """Declare the variables needed for derivation."""
        required = [
            {
                'short_name': 'rlds'
            },
            {
                'short_name': 'rlns'
            },
        ]
        return required

    @staticmethod
    def calculate(cubes):
        """Compute upwelling longwave flux from downwelling and net."""
        rlds_cube = cubes.extract_cube(
            Constraint(name='surface_downwelling_longwave_flux_in_air'))
        rlns_cube = cubes.extract_cube(
            Constraint(name='surface_net_downward_longwave_flux'))
        # fix latitude and longitude var_name
        rlns_cube.coord(axis='X').long_name = rlds_cube.coord(
            axis='X').long_name
        rlns_cube.coord(axis='Y').long_name = rlds_cube.coord(
            axis='Y').long_name
        rlns_cube.coord(axis='X').var_name = rlds_cube.coord(
            axis='X').var_name
        rlns_cube.coord(axis='Y').var_name = rlds_cube.coord(
            axis='Y').var_name

        rlus_cube = rlds_cube - rlns_cube

        rlus_cube.attributes['positive'] = 'up'

        return rlus_cube
