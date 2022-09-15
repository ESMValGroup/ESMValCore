"""Derivation of variable `sithick`."""

import dask.array as da

from iris import Constraint

from ._baseclass import DerivedVariableBase


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `siextent`."""

    @staticmethod
    def required(project):
        """Declare the variables needed for derivation."""
        required = [{
            'short_name': 'sic',
        }]
        return required

    @staticmethod
    def calculate(cubes):
        """
        Compute sea ice extent.

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

        sic = cubes.extract_cube(Constraint(name='sea_ice_area_fraction'))
        ones = da.ones_like(sic)
        siextent_data = da.ma.masked_where(sic.lazy_data() < 15., ones)
        siextent = sic.copy(siextent_data)
        siextent.units = 'm2'

        return siextent
