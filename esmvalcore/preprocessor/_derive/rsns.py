"""Derivation of variable `rsns`."""

from iris import Constraint

from ._baseclass import DerivedVariableBase


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `rsns`."""

    @staticmethod
    def required(project):  # noqa: ARG004
        """Declare the variables needed for derivation."""
        return [
            {"short_name": "rsds"},
            {"short_name": "rsus"},
        ]

    @staticmethod
    def calculate(cubes):
        """Compute surface net downward shortwave radiation."""
        rsds_cube = cubes.extract_cube(
            Constraint(name="surface_downwelling_shortwave_flux_in_air"),
        )
        rsus_cube = cubes.extract_cube(
            Constraint(name="surface_upwelling_shortwave_flux_in_air"),
        )

        return rsds_cube - rsus_cube
