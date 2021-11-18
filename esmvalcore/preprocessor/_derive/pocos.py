"""Derivation of variable `pocos`."""

from iris import Constraint

from ._baseclass import DerivedVariableBase


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `pocos`."""

    @staticmethod
    def required(project):
        """Declare the variables needed for derivation."""
        required = [
            {
                'short_name': 'detocos'
            },
            {
                'short_name': 'phycos'
            },
            {
                'short_name': 'zoocos'
            },
        ]
        return required

    @staticmethod
    def calculate(cubes):
        """Compute surface chlorophyll concentration."""
        detocos_cube = cubes.extract_cube(
            Constraint(name='mole_concentration_of_organic_' +
                            'detritus_expressed_as_carbon_in_sea_water'))
        phycos_cube = cubes.extract_cube(
            Constraint(name='mole_concentration_of_phytoplankton_' +
                            'expressed_as_carbon_in_sea_water'))
        zoocos_cube = cubes.extract_cube(
            Constraint(name='mole_concentration_of_zooplankton_' +
                            'expressed_as_carbon_in_sea_water'))
        pocos_cube = detocos_cube + phycos_cube + zoocos_cube

        return pocos_cube
