"""Derivation of variable `lwcre`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from iris import Constraint

from ._baseclass import DerivedVariableBase

if TYPE_CHECKING:
    from esmvalcore.typing import Facets


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `lwcre`."""

    @staticmethod
    def required(facets: Facets) -> list[Facets]:  # noqa: ARG004
        """Declare the variables needed for derivation."""
        return [
            {"short_name": "rlut"},
            {"short_name": "rlutcs"},
        ]

    @staticmethod
    def calculate(cubes):
        """Compute longwave cloud radiative effect."""
        rlut_cube = cubes.extract_cube(
            Constraint(name="toa_outgoing_longwave_flux"),
        )
        rlutcs_cube = cubes.extract_cube(
            Constraint(name="toa_outgoing_longwave_flux_assuming_clear_sky"),
        )

        lwcre_cube = rlutcs_cube - rlut_cube
        lwcre_cube.units = rlut_cube.units
        lwcre_cube.attributes["positive"] = "down"

        return lwcre_cube
