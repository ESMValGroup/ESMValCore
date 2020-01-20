"""Derivation of variable `nece`."""

import iris
import dask as da
from ._baseclass import DerivedVariableBase


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `nece` (Net Ecosystem Carbon Exchange)."""

    # Required variables
    required = [{'short_name': 'rh', },
                {'short_name': 'npp', }]
    # ra = Autotrophic respiration
    # npp = net primary production

    @staticmethod
    def calculate(cubes):
        """Compute Carbon Use Efficiency.
        """
        try:
            rh_cube = cubes.extract_strict(
                iris.Constraint(name='surface_upward_mass_flux_of_carbon_dioxide_expressed_as_carbon_due_to_heterotrophic_respiration'))
        except:
            print('\n\nFailed to extract ra from:', cubes )
            print('cube:', cubes[0].name(),
                  cubes[0].attributes['variant_label'],
                  cubes[0].attributes['experiment_id'])

        try:
            npp_cube = cubes.extract_strict(
                iris.Constraint(name='net_primary_productivity_of_biomass_expressed_as_carbon'))
        except: print('\n\nFailed to extract npp from:', cubes)

        return npp_cube - rh_cube
