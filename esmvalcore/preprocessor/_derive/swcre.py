"""Derivation of variable `swcre`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from iris import Constraint

from ._baseclass import DerivedVariableBase

if TYPE_CHECKING:
    from esmvalcore.typing import Facets


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `swcre`."""

    @staticmethod
    def required(facets: Facets) -> list[Facets]:  # noqa: ARG004
        """Declare the variables needed for derivation."""
        return [
            {"short_name": "rsut"},
            {"short_name": "rsutcs"},
        ]

    @staticmethod
    def calculate(cubes):
        """Compute shortwave cloud radiative effect."""
        rsut_cube = cubes.extract_cube(
            Constraint(name="toa_outgoing_shortwave_flux"),
        )
        rsutcs_cube = cubes.extract_cube(
            Constraint(name="toa_outgoing_shortwave_flux_assuming_clear_sky"),
        )

        swcre_cube = rsutcs_cube - rsut_cube
        swcre_cube.units = rsut_cube.units
        swcre_cube.attributes["positive"] = "down"

        return swcre_cube
