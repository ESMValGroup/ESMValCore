"""Derivation of variable `chlora`."""

from iris import Constraint

from ._baseclass import DerivedVariableBase


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `poc`."""

    @staticmethod
    def required(project):
        """Declare the variables needed for derivation."""
        required = [
            {
                'short_name': 'detoc'
            },
            {
                'short_name': 'phyc'
            },
            {
                'short_name': 'zooc'
            },
        ]
        return required

    @staticmethod
    def calculate(cubes):
        """Compute surface chlorophyll concentration."""
        detoc_cube = cubes.extract_cube(
            Constraint(name='mole_concentration_of_organic_' +
                            'detritus_expressed_as_carbon_in_sea_water'))
        phyc_cube = cubes.extract_cube(
            Constraint(name='mole_concentration_of_phytoplankton_' +
                            'expressed_as_carbon_in_sea_water'))
        zooc_cube = cubes.extract_cube(
            Constraint(name='mole_concentration_of_zooplankton_' +
                            'expressed_as_carbon_in_sea_water'))
        poc_cube = detoc_cube + phyc_cube + zooc_cube

        return poc_cube
