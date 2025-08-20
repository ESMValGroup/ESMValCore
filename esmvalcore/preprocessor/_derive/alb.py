"""Derivation of variable `alb`.

authors:
    - crez_ba

"""

from iris import NameConstraint

from ._baseclass import DerivedVariableBase


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `alb`."""

    @staticmethod
    def required(project):  # noqa: ARG004
        """Declare the variables needed for derivation."""
        return [
            {"short_name": "rsdscs"},
            {"short_name": "rsuscs"},
        ]

    @staticmethod
    def calculate(cubes):
        """Compute surface albedo."""
        rsdscs_cube = cubes.extract_cube(NameConstraint(var_name="rsdscs"))
        rsuscs_cube = cubes.extract_cube(NameConstraint(var_name="rsuscs"))

        return rsuscs_cube / rsdscs_cube
