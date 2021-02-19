"""Derivation of variable ``xch4``."""

from iris import Constraint

from ._baseclass import DerivedVariableBase
from ._shared import column_average


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable ``xch4``."""

    @staticmethod
    def required(project):
        """Declare the variables needed for derivation."""
        required = [
            {'short_name': 'ch4'},
            {'short_name': 'hus'},
            {'short_name': 'zg'},
            {'short_name': 'ps'},
        ]
        return required

    @staticmethod
    def calculate(cubes):
        """Calculate the column-averaged atmospheric CH4 [1e-9]."""
        ch4_cube = cubes.extract_cube(
            Constraint(name='mole_fraction_of_methane_in_air'))
        hus_cube = cubes.extract_cube(Constraint(name='specific_humidity'))
        zg_cube = cubes.extract_cube(Constraint(name='geopotential_height'))
        ps_cube = cubes.extract_cube(Constraint(name='surface_air_pressure'))

        # Column-averaged CH4
        xch4_cube = column_average(ch4_cube, hus_cube, zg_cube, ps_cube)
        xch4_cube.convert_units('1')

        return xch4_cube
