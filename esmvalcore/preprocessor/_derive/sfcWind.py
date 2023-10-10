"""Derivation of variable `sfcWind`."""

import cf_units
import iris
import numpy as np
from iris import NameConstraint

from ._baseclass import DerivedVariableBase

class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `sfcWind`."""

    @staticmethod
    def required(project):
        """Declare the variables needed for derivation."""
        required = [
            {
                'short_name': 'uas'
            },
            {
                'short_name': 'vas'
            },
        ]
        return required

    @staticmethod
    def calculate(cubes):
        """Compute surface near-surface wind speed 
           from eastward and northward components."""
        
        uas_cube = cubes.extract_cube(NameConstraint(var_name='uas'))
        vas_cube = cubes.extract_cube(NameConstraint(var_name='vas'))

        sqrt_f = iris.analysis.maths.IFunc(np.sqrt, 
                                           lambda cube: cf_units.Unit('m s-1'))

        sfcWind_cube = sqrt_f(uas_cube**2 + vas_cube**2)

        return sfcWind_cube