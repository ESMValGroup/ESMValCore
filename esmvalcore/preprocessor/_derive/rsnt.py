"""Derivation of variable `rsnt`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from iris import Constraint

from ._baseclass import DerivedVariableBase

if TYPE_CHECKING:
    from esmvalcore.typing import Facets


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `rsnt`."""

    @staticmethod
    def required(facets: Facets) -> list[Facets]:  # noqa: ARG004
        """Declare the variables needed for derivation."""
        return [
            {"short_name": "rsdt"},
            {"short_name": "rsut"},
        ]

    @staticmethod
    def calculate(cubes):
        """Compute toa net downward shortwave radiation."""
        rsdt_cube = cubes.extract_cube(
            Constraint(name="toa_incoming_shortwave_flux"),
        )
        rsut_cube = cubes.extract_cube(
            Constraint(name="toa_outgoing_shortwave_flux"),
        )

        rsnt_cube = rsdt_cube - rsut_cube
        rsnt_cube.units = rsdt_cube.units
        rsnt_cube.attributes["positive"] = "down"

        return rsnt_cube
