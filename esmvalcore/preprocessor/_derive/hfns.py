"""Derivation of variable `hfns`."""

from iris import NameConstraint

from ._baseclass import DerivedVariableBase


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `hfns`."""

    @staticmethod
    def required(project):
        """Declare the variables needed for derivation."""
        required = [
            {
                'short_name': 'hfls',
            },
            {
                'short_name': 'hfss',
            },
        ]
        return required

    @staticmethod
    def calculate(cubes):
        """Compute surface net heat flux."""
        hfls_cube = cubes.extract_cube(NameConstraint(var_name='hfls'))
        hfss_cube = cubes.extract_cube(NameConstraint(var_name='hfss'))

        hfns_cube = hfls_cube + hfss_cube
        hfns_cube.units = hfls_cube.units

        return hfns_cube
