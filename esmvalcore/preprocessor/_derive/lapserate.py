"""Derivation of variable ``lapserate``."""

import numpy as np

from iris import Constraint

from ._baseclass import DerivedVariableBase


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable ``lapserate``."""

    @staticmethod
    def required(project):  # noqa: ARG004
        """Declare the variables needed for derivation."""
        return [
            {"short_name": "ta"},
            {"short_name": "zg"},
        ]

    @staticmethod
    def calculate(cubes):
        """Calculate the lapse rates as vertical temperature gradient -dT/dz [K km-1]."""
        ta_cube = cubes.extract_cube(Constraint(name="air_temperature"))
        zg_cube = cubes.extract_cube(Constraint(name="geopotential_height"))

        # Lapse rate
        lapserate_cube = zg_cube.copy()
        lapserate_cube.data = (np.gradient(ta_cube.core_data(), axis=1) /
                              np.gradient(zg_cube.core_data(), axis=1) * -1000.)
        lapserate_cube.units="K km-1"

        return lapserate_cube
