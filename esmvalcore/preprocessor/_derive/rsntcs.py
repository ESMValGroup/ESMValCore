"""Derivation of variable `rsntcs`."""

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
            {"short_name": "rsutcs"},
        ]

    @staticmethod
    def calculate(cubes):
        """Compute toa net downward shortwave radiation assuming clear sky."""
        rsdt_cube = cubes.extract_cube(
            Constraint(name="toa_incoming_shortwave_flux"),
        )
        rsutcs_cube = cubes.extract_cube(
            Constraint(name="toa_outgoing_shortwave_flux_assuming_clear_sky"),
        )
        rsntcs_cube = rsdt_cube - rsutcs_cube
        rsntcs_cube.attributes["positive"] = "down"
        return rsntcs_cube
