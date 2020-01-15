"""Derivation of variable `co2flux`."""

from iris import Constraint

from ._baseclass import DerivedVariableBase
from .._regrid import regrid
#from .._weighting.py import weighting_landsea_fraction


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `co2flux`."""

    @staticmethod
    def required(project):
        """Declare the variables needed for derivation."""
        required = [
            {
                'short_name': 'nbp', 'mip': 'Lmon'
            },
            {
                'short_name': 'fgco2', 'mip': 'Omon'
            },
        ]
        return required

    @staticmethod
    def calculate(cubes):
        """Compute total carbon flux from land and ocean."""
        nbp_cube = cubes.extract_strict(
            Constraint(name='nbp'))
        fgco2_cube = cubes.extract_strict(
            Constraint(name='fgco2'))

        # Regridd fgco2 to linear grid
        fgco2_cube = regrid(fgco2_cube, nbp_cube, 'linear')

        # Account for leand-sea fraction (not working yet)
        #nbp_cube = weighting_landsea_fraction(nbp_cube, sftlf, land)
        #fgco2_cube = weighting_landsea_fraction(fgco2_cube, sftof, land)

        co2flux_cube = nbp_cube + fgco2_cube

        return co2flux_cube
