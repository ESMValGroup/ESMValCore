"""Derivation of variable `chlora`."""

from iris import Constraint

from ._baseclass import DerivedVariableBase


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `chlora`."""

    @staticmethod
    def required(project):  # noqa: ARG004
        """Declare the variables needed for derivation."""
        return [
            {"short_name": "chldiatos"},
            {"short_name": "chlmiscos"},
        ]

    @staticmethod
    def calculate(cubes):
        """Compute surface chlorophyll concentration."""
        chldiatos_cube = cubes.extract_cube(
            Constraint(
                name=(
                    "mass_concentration_of_diatoms_expressed_as"
                    "_chlorophyll_in_sea_water"
                ),
            ),
        )
        chlmiscos_cube = cubes.extract_cube(
            Constraint(
                name=(
                    "mass_concentration_of_miscellaneous"
                    "_phytoplankton_expressed_as_chlorophyll"
                    "_in_sea_water"
                ),
            ),
        )

        return chldiatos_cube + chlmiscos_cube
