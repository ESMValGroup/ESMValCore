"""Derivation of variable `wue`."""

import iris
import dask as da
from ._baseclass import DerivedVariableBase


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `wue` (Water Use Efficiency)."""

    # Required variables
    required = [{'short_name': 'gpp'},
                {'short_name': 'tran'}]

    # gpp = gross primary production
    # tran = Transpiration

    @staticmethod
    def calculate(cubes):
        """Compute Water Use Efficiency.
        """
        gpp_cube = cubes.extract_strict(
            iris.Constraint(name='gross_primary_productivity_of_biomass_expressed_as_carbon'))

        tran_cube = cubes.extract_strict(
            iris.Constraint(name='transpiration_flux'))

        new_cube = tran_cube/gpp_cube
        selection = (da.array.fabs(new_cube.data) <= 1000.)

        selection = da.array.broadcast_to(selection, new_cube.shape)
        new_cube.data = da.array.ma.masked_where(~selection, new_cube.core_data())

        return new_cube
