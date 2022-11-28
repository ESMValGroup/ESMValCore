"""Derivation of variable `sithick`."""
import logging

from iris import Constraint

from ._baseclass import DerivedVariableBase
from ._shared import rotate_vector
from .._regrid import regrid

logger = logging.getLogger(__name__)


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `siextent`."""

    @staticmethod
    def required(project):
        """Declare the variables needed for derivation."""
        required = [
            {
                'short_name': 'uo',
            },
            {
                'short_name': 'vo',
            },
            {
                'short_name': 'areacello',
            }]
        return required

    @staticmethod
    def calculate(cubes):
        """Compute .

        Arguments
        ---------
            cubes: cubelist containing .

        Returns
        -------
            Cube containing .
        """
        uo = cubes.extract_cube(Constraint(name='sea_water_x_velocity'))
        vo = cubes.extract_cube(Constraint(name='sea_water_y_velocity'))
        areacello = cubes.extract_cube(Constraint(name='cell_area'))

        urot = rotate_vector(uo, vo, 'x')
        urot = regrid(urot, areacello, 'linear')

        return urot
