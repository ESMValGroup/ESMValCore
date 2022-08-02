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

        Returns an array of ones in every grid point where the sea ice area fraction
        has values > 15 .

        Use in combination with the preprocessor `area_statistics(operator='sum')` to
        weigh by the area and compute global or regional sea ice extent values.

        Arguments
        ---------
            cubes: cubelist containing volume and concentration components.

        Returns
        -------
            Cube containing sea ice speed.

        """
        
        siconc = cubes.extract_cube(Constraint(name='sea_ice_area_fraction'))
        siconc.convert_units('1')
        ones = da.ones_like(siconc)
        siextent_data = da.ma.masked_where(siconc.lazy_data()>0.15, ones)
        siextent = siconc.copy(siextent_data)

        return siextent
