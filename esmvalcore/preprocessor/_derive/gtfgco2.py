"""Derivation of variable `gtfgco2`."""
import iris

from ._baseclass import DerivedVariableBase


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `gtfgco2`."""

    # Required variables
    required = [
        {
            'short_name': 'fgco2',
            'mip': 'Omon',
            'fx_files': [
                'areacello',
            ],
        },
    ]

    @staticmethod
    def calculate(cubes):
        """Compute longwave cloud radiative effect."""
        fgco2_cube = cubes.extract_strict(
            iris.Constraint(name='surface_downward_mass_flux_of_carbon_dioxide'
                            '_expressed_as_carbon'))
        area_cube = cubes.extract_strict(iris.Constraint(name='cell_area'))

        total_flux = (fgco2_cube * area_cube).collapsed(
            ['latitude', 'longitude'],
            iris.analysis.SUM,
        )

        return total_flux
