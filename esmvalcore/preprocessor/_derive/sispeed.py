"""Derivation of variable `sispeed`."""

import logging
from iris import Constraint

from .._regrid import regrid

from ._baseclass import DerivedVariableBase

logger = logging.getLogger(__name__)


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `sispeed`."""

    @staticmethod
    def required(project):
        """Declare the variables needed for derivation."""
        if project == 'CMIP6':
            required = [{'short_name': 'siu'}, {'short_name': 'siv'}]
        else:
            required = [{'short_name': 'usi'}, {'short_name': 'vsi'}]
        return required

    @staticmethod
    def calculate(cubes):
        """
        Compute sispeed module from velocity components siu and siv.

        Arguments
        ---------
            cubes: cubelist containing velocity components.

        Returns
        -------
            Cube containing sea ice speed.

        """
        siu = cubes.extract_cube(Constraint(name='sea_ice_x_velocity'))
        siv = cubes.extract_cube(Constraint(name='sea_ice_y_velocity'))
        try:
            return DerivedVariable._get_speed(siu, siv)
        except ValueError:
            logger.debug('Regridding siv into siu grid to compute sispeed')
            siv = regrid(siv, siu, 'linear')
            return DerivedVariable._get_speed(siu, siv)

    @staticmethod
    def _get_speed(siu, siv):
        return (siu**2 + siv**2)**0.5
