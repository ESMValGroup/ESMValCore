"""Derivation of variable `rsnst`.

authors:
    - weig_ka

"""

from __future__ import annotations

from typing import TYPE_CHECKING

from iris import Constraint

from ._baseclass import DerivedVariableBase

if TYPE_CHECKING:
    from esmvalcore.typing import Facets


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `rsnst`."""

    @staticmethod
    def required(facets: Facets) -> list[Facets]:  # noqa: ARG004
        """Declare the variables needed for derivation."""
        return [
            {"short_name": "rsds"},
            {"short_name": "rsdt"},
            {"short_name": "rsus"},
            {"short_name": "rsut"},
        ]

    @staticmethod
    def calculate(cubes):
        """Compute Heating from Shortwave Absorption."""
        rsds_cube = cubes.extract_cube(
            Constraint(name="surface_downwelling_shortwave_flux_in_air"),
        )
        rsdt_cube = cubes.extract_cube(
            Constraint(name="toa_incoming_shortwave_flux"),
        )
        rsus_cube = cubes.extract_cube(
            Constraint(name="surface_upwelling_shortwave_flux_in_air"),
        )
        rsut_cube = cubes.extract_cube(
            Constraint(name="toa_outgoing_shortwave_flux"),
        )

        return (rsdt_cube - rsut_cube) - (rsds_cube - rsus_cube)
