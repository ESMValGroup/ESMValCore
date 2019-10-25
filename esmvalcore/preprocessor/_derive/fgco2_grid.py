"""Derivation of variable `fgco2_grid`."""
from ._baseclass import DerivedVariableBase
from ._shared import grid_area_correction


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `fgco2_grid`."""

    @staticmethod
    def required(project):
        """Declare the variables needed for derivation."""
        required = [
            {
                'short_name': 'fgco2'
            },
            {
                'short_name': 'sftlf',
                'mip': 'fx',
                'optional': True
            },
            {
                'short_name': 'sftof',
                'mip': 'fx',
                'optional': True
            },
        ]
        if project == 'CMIP6':
            required = [
                {
                    'short_name': 'fgco2'
                },
                {
                    'short_name': 'sftlf',
                    'mip': 'fx',
                    'optional': True
                },
                {
                    'short_name': 'sftof',
                    'mip': 'Ofx',
                    'optional': True
                },
            ]
        return required

    @staticmethod
    def calculate(cubes):
        """Compute gas exchange flux of CO2 relative to grid cell area.

        Note
        ----
        By default, `fgco2` is defined relative to sea area. For spatial
        integration, the original quantity is multiplied by the sea area
        fraction (`sftof`), so that the resuting derived variable is defined
        relative to the grid cell area. This correction is only relevant for
        coastal regions.

        """
        return grid_area_correction(cubes, 'fgco2', ocean_var=True)
