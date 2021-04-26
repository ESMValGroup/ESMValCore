"""Derivation of variable ``xco2``."""

from iris import Constraint

from ._baseclass import DerivedVariableBase
from ._shared import column_average


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable ``xco2``."""

    @staticmethod
    def required(project):
        """Declare the variables needed for derivation."""
        required = [
            {'short_name': 'co2'},
            {'short_name': 'hus'},
            {'short_name': 'zg'},
            {'short_name': 'ps'},
        ]
        return required

    @staticmethod
    def calculate(cubes):
        """Calculate the column-averaged atmospheric CO2 [1e-6]."""
        co2_cube = cubes.extract_cube(
            Constraint(name='mole_fraction_of_carbon_dioxide_in_air'))
        print(co2_cube)
        hus_cube = cubes.extract_cube(Constraint(name='specific_humidity'))
        zg_cube = cubes.extract_cube(Constraint(name='geopotential_height'))
        ps_cube = cubes.extract_cube(Constraint(name='surface_air_pressure'))

        # Column-averaged CO2
        xco2_cube = column_average(co2_cube, hus_cube, zg_cube, ps_cube)
        xco2_cube.convert_units('1')

        return xco2_cube
