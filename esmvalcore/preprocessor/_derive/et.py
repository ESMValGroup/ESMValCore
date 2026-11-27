"""Derivation of variable `et`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cf_units
from iris import Constraint

from ._baseclass import DerivedVariableBase

if TYPE_CHECKING:
    from esmvalcore.typing import Facets

# Constants
LATENT_HEAT_VAPORIZATION = 2.465e6


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `et`."""

    @staticmethod
    def required(facets: Facets) -> list[Facets]:  # noqa: ARG004
        """Declare the variables needed for derivation."""
        return [{"short_name": "hfls", "mip": "Amon"}]

    @staticmethod
    def calculate(cubes):
        """Compute evapotranspiration."""
        hfls_cube = cubes.extract_cube(
            Constraint(name="surface_upward_latent_heat_flux"),
        )

        et_cube = hfls_cube * 24.0 * 3600.0 / LATENT_HEAT_VAPORIZATION
        et_cube.units = cf_units.Unit("mm day-1")
        et_cube.attributes.pop("positive", None)

        return et_cube
