"""Derivation of variable `sfcWind`."""

from iris import NameConstraint

from ._baseclass import DerivedVariableBase


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `sfcWind`."""

    @staticmethod
    def required(project):  # noqa: ARG004
        """Declare the variables needed for derivation."""
        return [
            {"short_name": "uas"},
            {"short_name": "vas"},
        ]

    @staticmethod
    def calculate(cubes):
        """Compute near-surface wind speed.

        Wind speed derived from eastward and northward components.
        """
        uas_cube = cubes.extract_cube(NameConstraint(var_name="uas"))
        vas_cube = cubes.extract_cube(NameConstraint(var_name="vas"))

        return (uas_cube**2 + vas_cube**2) ** 0.5
