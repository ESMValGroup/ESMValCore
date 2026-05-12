"""Derivation of variable `clltkisccp`."""

from iris import Constraint

from ._baseclass import DerivedVariableBase
from ._shared import cloud_area_fraction


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `clltkisccp`."""

    @staticmethod
    def required(project):  # noqa: ARG004
        """Declare the variables needed for derivation."""
        return [{"short_name": "clisccp"}]

    @staticmethod
    def calculate(cubes):
        """Compute ISCCP low level thick cloud area fraction."""
        tau = Constraint(
            atmosphere_optical_thickness_due_to_cloud=lambda t: t > 23.0,
        )
        plev = Constraint(air_pressure=lambda p: p > 68000.0)

        return cloud_area_fraction(cubes, tau, plev)
