"""Derivation of variable `alb`.

authors:
    - crez_ba

"""

from __future__ import annotations

from typing import TYPE_CHECKING

from iris import NameConstraint

from ._baseclass import DerivedVariableBase

if TYPE_CHECKING:
    from esmvalcore.typing import Facets


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `alb`."""

    @staticmethod
    def required(facets: Facets) -> list[Facets]:  # noqa: ARG004
        """Declare the variables needed for derivation."""
        return [
            {"short_name": "rsdscs"},
            {"short_name": "rsuscs"},
        ]

    @staticmethod
    def calculate(cubes):
        """Compute surface albedo."""
        rsdscs_cube = cubes.extract_cube(NameConstraint(var_name="rsdscs"))
        rsuscs_cube = cubes.extract_cube(NameConstraint(var_name="rsuscs"))

        return rsuscs_cube / rsdscs_cube
