"""Derivation of variable `hfns`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from iris import NameConstraint

from ._baseclass import DerivedVariableBase

if TYPE_CHECKING:
    from esmvalcore.typing import Facets


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `hfns`."""

    @staticmethod
    def required(facets: Facets) -> list[Facets]:  # noqa: ARG004
        """Declare the variables needed for derivation."""
        return [
            {
                "short_name": "hfls",
            },
            {
                "short_name": "hfss",
            },
        ]

    @staticmethod
    def calculate(cubes):
        """Compute surface net heat flux."""
        hfls_cube = cubes.extract_cube(NameConstraint(var_name="hfls"))
        hfss_cube = cubes.extract_cube(NameConstraint(var_name="hfss"))

        hfns_cube = hfls_cube + hfss_cube
        hfns_cube.units = hfls_cube.units

        return hfns_cube
