"""Derivation of variable `cue`."""

import iris
import dask as da
from ._baseclass import DerivedVariableBase


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `cue` (Carbon Use Efficiency)."""

    # Required variables
    required = [{'short_name': 'gpp', },
                {'short_name': 'npp', }]
    # npp = net primary production
    # gpp = gross primary production

    @staticmethod
    def calculate(cubes):
        """Compute Carbon Use Efficiency.
        """
        try:
            gpp_cube = cubes.extract_strict(
                iris.Constraint(name='gross_primary_productivity_of_biomass_expressed_as_carbon'))
        except:
            print('\n\nFailed to extract gpp from :', cubes )
            print('cube:', cubes[0].name(),
                  cubes[0].attributes['variant_label'],
                  cubes[0].attributes['experiment_id'])

        try:
            npp_cube = cubes.extract_strict(
                iris.Constraint(name='net_primary_productivity_of_biomass_expressed_as_carbon'))
        except: print('\n\nFailed to extract npp from :', cubes)

        new_cube = npp_cube/gpp_cube
        selection = (da.array.fabs(new_cube.data) <= 1.)

        selection = da.array.broadcast_to(selection, new_cube.shape)
        new_cube.data = da.array.ma.masked_where(~selection, new_cube.core_data())

        #print (new_cube.data.min(), new_cube.data.max())

        return new_cube
