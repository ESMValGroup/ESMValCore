"""Derivation of variable `rtnt`."""

from iris import Constraint

from ._baseclass import DerivedVariableBase


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `rtnt`."""

    @staticmethod
    def required(project):  # noqa: ARG004
        """Declare the variables needed for derivation."""
        return [
            {"short_name": "rsdt"},
            {"short_name": "rsut"},
            {"short_name": "rlut"},
        ]

    @staticmethod
    def calculate(cubes):
        """Compute toa net downward total radiation."""
        rsdt_cube = cubes.extract_cube(
            Constraint(name="toa_incoming_shortwave_flux"),
        )
        rsut_cube = cubes.extract_cube(
            Constraint(name="toa_outgoing_shortwave_flux"),
        )
        rlut_cube = cubes.extract_cube(
            Constraint(name="toa_outgoing_longwave_flux"),
        )

        return rsdt_cube - rsut_cube - rlut_cube
