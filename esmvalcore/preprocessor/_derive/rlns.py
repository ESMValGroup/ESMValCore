"""Derivation of variable `rlns`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from iris import Constraint

from ._baseclass import DerivedVariableBase

if TYPE_CHECKING:
    from esmvalcore.typing import Facets


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `rlns`."""

    @staticmethod
    def required(facets: Facets) -> list[Facets]:  # noqa: ARG004
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
