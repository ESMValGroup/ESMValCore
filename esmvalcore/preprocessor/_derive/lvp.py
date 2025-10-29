"""Derivation of variable `lvp`.

authors:
    - weig_ka

"""

from iris import NameConstraint

from ._baseclass import DerivedVariableBase


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `lvp`."""

    @staticmethod
    def required(project):  # noqa: ARG004
        """Declare the variables needed for derivation."""
        return [
            {"short_name": "hfls"},
            {"short_name": "pr"},
            {"short_name": "evspsbl"},
        ]

    @staticmethod
    def calculate(cubes):
        """Compute Latent Heat Release from Precipitation."""
        hfls_cube = cubes.extract_cube(NameConstraint(var_name="hfls"))
        pr_cube = cubes.extract_cube(NameConstraint(var_name="pr"))
        evspsbl_cube = cubes.extract_cube(NameConstraint(var_name="evspsbl"))

        return hfls_cube * (pr_cube / evspsbl_cube)
