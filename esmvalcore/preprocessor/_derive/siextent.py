"""Derivation of variable `sithick`."""
import logging

import dask.array as da
import iris
from iris import Constraint

from esmvalcore.exceptions import RecipeError
from ._baseclass import DerivedVariableBase

logger = logging.getLogger(__name__)


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `siextent`."""

    @staticmethod
    def required(project):
        """Declare the variables needed for derivation."""
        required = [
            {
                'short_name': 'sic',
                'optional': 'true'
            },
            {
                'short_name': 'siconca',
                'optional': 'true'
            }]
        return required

    @staticmethod
    def calculate(cubes):
        """Compute sea ice extent.

        Returns an array of ones in every grid point where
        the sea ice area fraction has values > 15 .

        Use in combination with the preprocessor
        `area_statistics(operator='sum')` to weigh by the area and
        compute global or regional sea ice extent values.

        Arguments
        ---------
            cubes: cubelist containing sea ice area fraction.

        Returns
        -------
            Cube containing sea ice extent.
        """
        try:
            sic = cubes.extract_cube(Constraint(name='sic'))
        except iris.exceptions.ConstraintMismatchError:
            try:
                sic = cubes.extract_cube(Constraint(name='siconca'))
            except iris.exceptions.ConstraintMismatchError as exc:
                raise RecipeError(
                    'Derivation of siextent failed due to missing variables '
                    'sic and siconca.') from exc

        ones = da.ones_like(sic)
        siextent_data = da.ma.masked_where(sic.lazy_data() < 15., ones)
        siextent = sic.copy(siextent_data)
        siextent.units = 'm2'

        return siextent
