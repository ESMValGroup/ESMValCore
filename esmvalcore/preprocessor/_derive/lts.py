"""Derivation of variable `lts` (lower tropospheric stability)."""

import logging

import iris
from esmvalcore.iris_helpers import var_name_constraint

from ._baseclass import DerivedVariableBase

logger = logging.getLogger(__name__)


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `lts` (lower tropospheric stability)."""

    @staticmethod
    def required(project):
        """Declare the variables needed for derivation."""
        required = [
            {
                'short_name': 'ta'
            },
        ]
        return required

    @staticmethod
    def calculate(cubes):
        """Compute lower tropospheric stability.

        lts = theta(700 hPa) - theta(surface)

        potential temperature theta = T * (p_0 / p) ^(R / cp)

        with

        T = temperature (K)
        p_0 = 1000 hPa
        p = pressure (hPa)
        R = gas constant for air (= 8.314 / 28.96e-3 = 287.058 J kg-1 K-1)
        cp = specific heat capacity of air at constant pressure
             (= 1006 J kg-1 K-1)
        R / cp = 0.286

        """
        ta = cubes.extract_cube(var_name_constraint('ta'))

        t1000 = ta.interpolate([('air_pressure', 100000.)],
                               scheme=iris.analysis.Linear())
        t700 = ta.interpolate([('air_pressure', 70000.)],
                               scheme=iris.analysis.Linear())

        theta1000 = t1000
        theta700 = t700 * (100000. / 70000.) ** 0.286

        lts = theta700 - theta1000

        return lts
