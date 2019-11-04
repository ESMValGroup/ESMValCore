"""Derivation of variable `lai_grid`."""

from ._baseclass import DerivedVariableBase
from ._shared import grid_area_correction


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `lai_grid`."""

    @staticmethod
    def required(project):
        """Declare the variables needed for derivation."""
        required = [
            {
                'short_name': 'lai'
            },
            {
                'short_name': 'sftlf',
                'mip': 'fx',
                'optional': True
            },
        ]
        return required

    @staticmethod
    def calculate(cubes):
        """Compute leaf area index relative to grid cell area.

        Note
        ----
        By default, `lai` is defined relative to land area. For spatial
        integration, the original quantity is multiplied by the land area
        fraction (`sftlf`), so that the resuting derived variable is defined
        relative to the grid cell area. This correction is only relevant for
        coastal regions.

        """
        return grid_area_correction(cubes, 'lai')
