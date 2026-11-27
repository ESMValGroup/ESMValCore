"""Derivation of variable `ctotal`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import iris
from iris import Constraint

from ._baseclass import DerivedVariableBase

if TYPE_CHECKING:
    from esmvalcore.typing import Facets


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `ctotal`."""

    # Required variables
    @staticmethod
    def required(facets: Facets) -> list[Facets]:
        """Declare the variables needed for derivation."""
        project = facets["project"]
        required: list[Facets]
        if project == "CMIP5":
            required = [
                {"short_name": "cVeg", "mip": "Lmon"},
                {"short_name": "cSoil", "mip": "Lmon"},
            ]
        elif project == "CMIP6":
            required = [
                {"short_name": "cVeg", "mip": "Lmon"},
                {"short_name": "cSoil", "mip": "Emon"},
            ]
        elif project == "CMIP7":
            required = [
                {
                    "short_name": "cVeg",
                    "branding_suffix": "tavg-u-hxy-lnd",
                    "mip": "land",
                },
                {
                    "short_name": "cSoil",
                    "branding_suffix": "tavg-u-hxy-lnd",
                    "mip": "land",
                },
            ]
        else:
            msg = (
                f"Project {project} cannot be used for derivation of 'ctotal'"
            )
            raise ValueError(msg)
        return required

    @staticmethod
    def calculate(cubes):
        """Compute total ecosystem carbon storage."""
        c_soil_cube = cubes.extract_cube(
                Constraint(varname="cSoil"),
            )
        c_veg_cube = cubes.extract_cube(
            Constraint(varname="cVeg"),
        )
        c_total_cube = c_soil_cube + c_veg_cube
        c_total_cube.standard_name = None
        c_total_cube.long_name = "Total Carbon Stock"
        return c_total_cube
