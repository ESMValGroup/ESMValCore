"""Derivation of variable `sispeed`."""

from iris import Constraint
from iris.coords import DimCoord

from ._baseclass import DerivedVariableBase


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `sispeed`."""

    # Required variables
    required = [
        {'short_name': 'usi', },
        {'short_name': 'vsi', }
    ]

    @staticmethod
    def calculate(cubes):
        """
        Compute sispeed module from velocity components siu and siv.

        Arguments
        ----
            cubes: cubelist containing velocity components.

        Returns
        -------
            Cube containing sea ice speed.

        """
        siu = cubes.extract_strict(Constraint(name='sea_ice_x_velocity'))

        siv = cubes.extract_strict(Constraint(name='sea_ice_y_velocity'))
        siv.remove_coord('latitude')
        siv.remove_coord('longitude')

        # Models usually store siu and siv in slightly different points
        if isinstance(siu.coord('latitude'), DimCoord):
            siv.add_dim_coord(
                siu.coord('latitude'), siu.coord_dims('latitude')
            )
            siv.add_dim_coord(
                siu.coord('longitude'), siu.coord_dims('longitude')
            )
        else:
            siv.add_aux_coord(
                siu.coord('latitude'), siu.coord_dims('latitude')
            )
            siv.add_aux_coord(
                siu.coord('longitude'), siu.coord_dims('longitude')
            )

        speed = (siu ** 2 + siv ** 2) ** 0.5
        return speed
