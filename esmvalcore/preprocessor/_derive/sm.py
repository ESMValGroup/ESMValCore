"""Derivation of variable `sm`."""

import cf_units
import numpy as np

from esmvalcore.iris_helpers import var_name_constraint

from ._baseclass import DerivedVariableBase


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `sm`."""

    @staticmethod
    def required(project):
        """Declare the variables needed for derivation."""
        required = [{'short_name': 'mrsos'}]
        return required

    @staticmethod
    def calculate(cubes):
        """Compute soil moisture.

        Note
        ----
        Convert moisture content of soil layer (kg/m2) into volumetric soil
        moisture (m3/m3), assuming density of water 998.2 kg/m2 (at temperature
        20 deg C).

        """
        mrsos_cube = cubes.extract_cube(var_name_constraint('mrsos'))

        depth = mrsos_cube.coord('depth').bounds.astype(np.float32)
        layer_thickness = depth[..., 1] - depth[..., 0]

        sm_cube = mrsos_cube / layer_thickness / 998.2
        sm_cube.units = cf_units.Unit('m3 m^-3')

        return sm_cube
