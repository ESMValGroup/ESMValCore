"""Derivation of variable `rlns`."""

from iris import Constraint

from ._baseclass import DerivedVariableBase


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `rlns`."""

    @staticmethod
    def required(project):  # noqa: ARG004
        """Declare the variables needed for derivation."""
        return [
            {"short_name": "rlds"},
            {"short_name": "rlus"},
        ]

    @staticmethod
    def calculate(cubes):
        """Compute surface net downward longwave radiation."""
        rlds_cube = cubes.extract_cube(
            Constraint(name="surface_downwelling_longwave_flux_in_air"),
        )
        rlus_cube = cubes.extract_cube(
            Constraint(name="surface_upwelling_longwave_flux_in_air"),
        )

        return rlds_cube - rlus_cube
