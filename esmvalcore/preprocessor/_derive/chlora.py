"""Derivation of variable `chlora`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from iris import Constraint

from ._baseclass import DerivedVariableBase

if TYPE_CHECKING:
    from esmvalcore.typing import Facets


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `chlora`."""

    @staticmethod
    def required(facets: Facets) -> list[Facets]:  # noqa: ARG004
        """Declare the variables needed for derivation."""
        project = facets["project"]
        if project == "CMIP7":
            return [
                {
                    "short_name": "chldiat",
                    "branding_suffix": "tavg-ols-hxy-seamsftm",
                    "mip": "ocnBgchem",
                },
                {
                    "short_name": "chlmisc",
                    "branding_suffix": "tavg-ols-hxy-sea",
                    "mip": "ocnBgchem",
                },
            ]
        else:
            return [
                {"short_name": "chldiatos"},
                {"short_name": "chlmiscos"},
            ]

    @staticmethod
    def calculate(cubes):
        """Compute surface chlorophyll concentration."""
        chldiatos_cube = cubes.extract_cube(
            Constraint(
                name=(
                    "mass_concentration_of_diatoms_expressed_as"
                    "_chlorophyll_in_sea_water"
                ),
            ),
        )
        chlmiscos_cube = cubes.extract_cube(
            Constraint(
                name=(
                    "mass_concentration_of_miscellaneous"
                    "_phytoplankton_expressed_as_chlorophyll"
                    "_in_sea_water"
                ),
            ),
        )

        return chldiatos_cube + chlmiscos_cube
