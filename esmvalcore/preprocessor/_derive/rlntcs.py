"""Derivation of variable `rlntcs`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from iris import Constraint

from ._baseclass import DerivedVariableBase

if TYPE_CHECKING:
    from esmvalcore.typing import Facets


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `rlntcs`."""

    @staticmethod
    def required(facets: Facets) -> list[Facets]:  # noqa: ARG004
        """Declare the variables needed for derivation."""
        return [{"short_name": "rlutcs"}]

    @staticmethod
    def calculate(cubes):
        """Compute toa net downward longwave radiation assuming clear sky."""
        rlutcs_cube = cubes.extract_cube(
            Constraint(name="toa_outgoing_longwave_flux_assuming_clear_sky"),
        )
        rlutcs_cube.data = -rlutcs_cube.core_data()
        rlutcs_cube.attributes["positive"] = "down"
        return rlutcs_cube
