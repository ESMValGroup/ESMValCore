"""Derivation of variable `ctotal`."""

import iris

from iris import Constraint

from ._baseclass import DerivedVariableBase


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `ctotal`."""

    # Required variables
    @staticmethod
    def required(project):
        """Declare the variables needed for derivation."""
        if project == 'CMIP5':
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
        elif project == 'CMIP6':
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
        return required

    @staticmethod
    def calculate(cubes):
        """Compute total ecosystem carbon storage."""
        try:
            c_soil_cube = cubes.extract_cube(
                Constraint(name='soil_carbon_content'))
        except iris.exceptions.ConstraintMismatchError:
            try:
                c_soil_cube = cubes.extract_cube(
                    Constraint(name='soil_mass_content_of_carbon'))
            except iris.exceptions.ConstraintMismatchError:
                raise ValueError(f"No cube from {cubes} can be loaded with "
                                 f"standard name CMIP5: soil_carbon_content "
                                 f"or CMIP6: soil_mass_content_of_carbon")
        c_veg_cube = cubes.extract_cube(
            Constraint(name='vegetation_carbon_content'))
        c_total_cube = c_soil_cube + c_veg_cube
        c_total_cube.standard_name = None
        c_total_cube.long_name = 'Total Carbon Stock'
        return c_total_cube
