"""Derivation of variable `vegFrac`."""

from esmvalcore.iris_helpers import var_name_constraint

from ._baseclass import DerivedVariableBase


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `vegFrac`."""

    @staticmethod
    def required(project):
        """Declare the variables needed for derivation."""
        required = [
            {'short_name': 'baresoilFrac'},
            {'short_name': 'residualFrac'},
        ]
        return required

    @staticmethod
    def calculate(cubes):
        """Compute vegetation fraction from bare soil fraction."""
        baresoilfrac_cube = cubes.extract_strict(var_name_constraint(
            'baresoilFrac'))
        residualfrac = cubes.extract_strict(var_name_constraint(
            'residualFrac'))

        baresoilfrac_cube.data = (100.0 - baresoilfrac_cube.core_data() -
                                  residualfrac.core_data())
        return baresoilfrac_cube
