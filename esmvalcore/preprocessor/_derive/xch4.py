"""Derivation of variable ``xch4``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from iris import Constraint

from ._baseclass import DerivedVariableBase
from ._shared import column_average

if TYPE_CHECKING:
    from esmvalcore.typing import Facets


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable ``xch4``."""

    @staticmethod
    def required(facets: Facets) -> list[Facets]:  # noqa: ARG004
        """Declare the variables needed for derivation."""
        return [
            {"short_name": "ch4"},
            {"short_name": "hus"},
            {"short_name": "zg"},
            {"short_name": "ps"},
        ]
        project = facets["project"]
        if project == "CMIP7":
            # For CMIP7, there are only three valid options for the branding_suffix of the
            # 3-dim variables: tavg-p19-hxy-air, tavg-p19-hxy-u and tavg-al-hxy-u.
            # These might differ among the three required 3-dim variables :(
            valid_options = ["tavg-p19-hxy-air", "tavg-p19-hxy-u", "tavg-al-hxy-u"]

            # check if branding_suffix was set in recipe
            # if not, we use the 19-level suffix ("p19")
            if "branding_suffix" in facets:
                suffix = facets["branding_suffix"]
            else:
                suffix = valid_options[0]

            # make sure branding_suffix is one of the valid options
            if suffix not in valid_options:
               suffix = valid_options[0]
            
            # last element of 19-level suffix "tavg-p19-hxy-xxx" differs for hus and ch4 / zg
            if suffix.split("-")[1] == "p19":
                # split suffix into elements (delimiter "-"), then make sure last element is "u" or "air" 
                u_suffix = "-".join([*suffix.split("-")[:-1], "u"])
                air_suffix = "-".join([*suffix.split("-")[:-1], "air"])
            else:
                u_suffix = suffix 
                air_suffix = suffix 

            return [
                {"short_name": "ch4", "branding_suffix": air_suffix, "mip": "atmosChem"},
                {"short_name": "hus", "branding_suffix": u_suffix, "mip": "atmos"},
                {"short_name": "zg", "branding_suffix": air_suffix, "mip": "atmos"},
                {"short_name": "ps", "branding_suffix": "tavg-u-hxy-u", "mip": "atmos"},
            ]
        else:
            return [
                {"short_name": "ch4"},
                {"short_name": "hus"},
                {"short_name": "zg"},
                {"short_name": "ps"},
            ]


    @staticmethod
    def calculate(cubes):
        """Calculate the column-averaged atmospheric CH4 [1e-9]."""
        ch4_cube = cubes.extract_cube(
            Constraint(name="mole_fraction_of_methane_in_air"),
        )
        hus_cube = cubes.extract_cube(Constraint(name="specific_humidity"))
        zg_cube = cubes.extract_cube(Constraint(name="geopotential_height"))
        ps_cube = cubes.extract_cube(Constraint(name="surface_air_pressure"))

        # Column-averaged CH4
        xch4_cube = column_average(ch4_cube, hus_cube, zg_cube, ps_cube)
        xch4_cube.convert_units("1")

        return xch4_cube
