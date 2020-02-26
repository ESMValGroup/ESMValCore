"""Derivation of variable `co2flux`."""

import iris

from ._baseclass import DerivedVariableBase
from ._shared import _var_name_constraint
from .._regrid import regrid
#from .._area import  area_statistics
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
            _var_name_constraint('nbp'))
        fgco2_cube = cubes.extract_strict(
            _var_name_constraint('fgco2'))

        # Regridd fgco2 to linear grid
        if fgco2_cube.shape[1] != 1 and fgco2_cube.shape[2] != 1:
            fgco2_cube = regrid(fgco2_cube, nbp_cube[0], 'linear')
        #nbp_cube   = area_statistics(nbp_cube,'sum')
        #fgco2_cube = area_statistics(fgco2_cube,'sum',fx_file!)


        # Account for leand-sea fraction (not working yet)
        #nbp_cube = weighting_landsea_fraction(nbp_cube, sftlf, land)
        #fgco2_cube = weighting_landsea_fraction(fgco2_cube, sftof, land)

        co2flux_cube = nbp_cube + fgco2_cube

        return co2flux_cube
