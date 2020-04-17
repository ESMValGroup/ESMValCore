"""Derivation of variable `ctotal`."""

from ._baseclass import DerivedVariableBase

from ._shared import var_name_constraint


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `ctotal`."""

    # Required variables
    @staticmethod
    def required(project):
        """Declare the variables needed for derivation."""
        if project == 'CMIP6':
            required = [
                {
                    'short_name': 'cVeg',
                    'mip': 'Lmon'
                },
                {
                    'short_name': 'cSoil',
                    'mip': 'Emon'
                },
            ]
        else:
            required = [
                {
                    'short_name': 'cVeg',
                    'mip': 'Lmon'
                },
                {
                    'short_name': 'cSoil',
                    'mip': 'Lmon'
                },
            ]

        return required

    @staticmethod
    def calculate(cubes):
        """Compute total ecosystem carbon storage."""
        c_soil_cube = cubes.extract_strict(var_name_constraint('cSoil'))
        c_veg_cube = cubes.extract_strict(var_name_constraint('cVeg'))
        c_total_cube = c_soil_cube + c_veg_cube
        c_total_cube.standard_name = None
        c_total_cube.long_name = 'Total Carbon Stock'
        return c_total_cube
