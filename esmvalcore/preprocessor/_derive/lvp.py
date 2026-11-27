"""Derivation of variable `lvp`.

authors:
    - weig_ka

"""

from __future__ import annotations

from typing import TYPE_CHECKING

from iris import NameConstraint

from ._baseclass import DerivedVariableBase

if TYPE_CHECKING:
    from esmvalcore.typing import Facets


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `lvp`."""

    @staticmethod
    def required(facets: Facets) -> list[Facets]:  # noqa: ARG004
        """Declare the variables needed for derivation."""
        return [
            {"short_name": "hfls"},
            {"short_name": "pr"},
            {"short_name": "evspsbl"},
        ]

    @staticmethod
    def calculate(cubes):
        """Compute Latent Heat Release from Precipitation."""
        hfls_cube = cubes.extract_cube(NameConstraint(var_name="hfls"))
        pr_cube = cubes.extract_cube(NameConstraint(var_name="pr"))
        evspsbl_cube = cubes.extract_cube(NameConstraint(var_name="evspsbl"))

        return hfls_cube * (pr_cube / evspsbl_cube)
