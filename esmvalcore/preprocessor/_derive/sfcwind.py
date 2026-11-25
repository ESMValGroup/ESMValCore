"""Derivation of variable `sfcWind`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from iris import NameConstraint

from ._baseclass import DerivedVariableBase

if TYPE_CHECKING:
    from esmvalcore.typing import Facets


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `sfcWind`."""

    @staticmethod
    def required(facets: Facets) -> list[Facets]:  # noqa: ARG004
        """Declare the variables needed for derivation."""
        return [
            {"short_name": "uas"},
            {"short_name": "vas"},
        ]

    @staticmethod
    def calculate(cubes):
        """Compute near-surface wind speed.

        Wind speed derived from eastward and northward components.
        """
        uas_cube = cubes.extract_cube(NameConstraint(var_name="uas"))
        vas_cube = cubes.extract_cube(NameConstraint(var_name="vas"))

        return (uas_cube**2 + vas_cube**2) ** 0.5
