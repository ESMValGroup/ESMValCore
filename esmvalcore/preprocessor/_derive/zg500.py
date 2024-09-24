"""Derivation of variable ``zg500``."""

from iris import Constraint

from ._baseclass import DerivedVariableBase
from .._regrid import extract_levels

class DerivedVariable(DerivedVariableBase):
    """Derivation of variable ``zg500``."""

    @staticmethod
    def required(project):
        """Declare the variables needed for derivation."""
        required = [
            {'short_name': 'zg'},
        ]
        return required

    @staticmethod
    def calculate(cubes):
        """Calculate the column-averaged atmospheric CH4 [1e-9]."""
        zg_cube = cubes.extract_cube(
            Constraint(name='geopotential_height'))

        # extract 500 hPa
        levels = 50000 # in Pa

        coord_z = zg_cube.coord(axis='Z')

        # check units
        if coord_z.units == 'Pa':
            # do nothing
            pass
        elif coord_z.units == 'hPa':
            levels /= 100. # convert from Pa to hPa
        else:
            print('Levels units not compatible with cube units')
            print( coord_z.units )
            raise Exception

        zg500_cube = extract_levels( zg_cube, levels, scheme='linear', coordinate=coord_z )

        return zg500_cube
