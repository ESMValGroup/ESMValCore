"""Derivation of variable `ctotal`."""

from iris import Constraint

from ._baseclass import DerivedVariableBase


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `ctotal`."""

    # Required variables
    required = [
        {
            'short_name': 'cVeg'
        },
        {
            'short_name': 'cSoil'
        },
    ]

    @staticmethod
    def calculate(cubes):
        """Compute total ecosystem carbon storage."""
        c_soil_cube = cubes.extract_strict(
            Constraint(name='soil_carbon_content'))
        c_veg_cube = cubes.extract_strict(
            Constraint(name='vegetation_carbon_content'))
        c_total_cube = c_soil_cube + c_veg_cube
        c_total_cube.standard_name = None
        c_total_cube.long_name = 'Total Carbon Stock'
        return c_total_cube
