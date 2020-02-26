"""Derivation of variable `sispeed`."""

import logging
import numpy as np
from iris import Constraint
from iris.coords import DimCoord

from ._baseclass import DerivedVariableBase

logger = logging.getLogger(__name__)


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `sispeed`."""

    @staticmethod
    def required(project):
        """Declare the variables needed for derivation."""
        required = [{
            'short_name': 'usi',
        }, {
            'short_name': 'vsi',
        }]
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
        siu = cubes.extract_strict(Constraint(name='sea_ice_x_velocity'))
        siv = cubes.extract_strict(Constraint(name='sea_ice_y_velocity'))

        try:
            return DerivedVariable._get_speed(siu, siv)
        except ValueError:
            # Models usually store siu and siv in slightly different points
            if not DerivedVariable._coordinate_close(siu, siv, 'latitude'):
                raise
            if not DerivedVariable._coordinate_close(siu, siv, 'longitude'):
                raise
            logger.warning(
                'Coordinates for sea ice velocity components differ. '
                'Changing y component latitude and longitude to match the '
                'coordinates of x component')
            siv.remove_coord('latitude')
            siv.remove_coord('longitude')

            if isinstance(siu.coord('latitude'), DimCoord):
                siv.add_dim_coord(siu.coord('latitude'),
                                  siu.coord_dims('latitude'))
                siv.add_dim_coord(siu.coord('longitude'),
                                  siu.coord_dims('longitude'))
            else:
                siv.add_aux_coord(siu.coord('latitude'),
                                  siu.coord_dims('latitude'))
                siv.add_aux_coord(siu.coord('longitude'),
                                  siu.coord_dims('longitude'))
            return DerivedVariable._get_speed(siu, siv)

    @staticmethod
    def _coordinate_close(siu, siv, coord):
        return np.allclose(
            siu.coord(coord).points,
            siv.coord(coord).points,
            rtol=0.,
            atol=5.,
        )

    @staticmethod
    def _get_speed(siu, siv):
        return (siu**2 + siv**2)**0.5
