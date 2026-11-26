"""Derivation of variable ``xco2``."""

from iris import Constraint

from ._baseclass import DerivedVariableBase
from ._shared import column_average


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable ``xco2``."""

    @staticmethod
    def required(project):  # noqa: ARG004
        """Declare the variables needed for derivation."""
        if project == "CMIP7":
            org_suffix = None
            # if no suffix is given, use "tavg-p19-hxy-air" for 3-dim variables as a default
            if not org_suffix:
                org_suffix = "tavg-p19-hxy-air"
            # last element of suffix "tavg-p19-hxy-air" differs for hus (= "u")
            if org_suffix.split("-")[1] == "p19":
                # split suffix into elements (delimiter "-"), then change last element into "u" 
                mod_suffix = "-".join([org_suffix.split("-")[:-1], "-u"])
            else:
                mod_suffix = org_suffix 
            return [
                {"short_name": "co2", "branding_suffix": org_suffix},
                {"short_name": "hus", "branding_suffix": mod_suffix},
                {"short_name": "zg", "branding_suffix": org_suffix},
                # branding suffix for "ps" is different than for 3-dim variables "co2", "hus", "zg"
                {"short_name": "ps", "branding_suffix": "tavg-u-hxy-u"},
            ]
        else:
            return [
                {"short_name": "co2"},
                {"short_name": "hus"},
                {"short_name": "zg"},
                {"short_name": "ps"},
            ]

    @staticmethod
    def calculate(cubes):
        """Calculate the column-averaged atmospheric CO2 [1e-6]."""
        co2_cube = cubes.extract_cube(
            Constraint(name="mole_fraction_of_carbon_dioxide_in_air"),
        )
        hus_cube = cubes.extract_cube(Constraint(name="specific_humidity"))
        zg_cube = cubes.extract_cube(Constraint(name="geopotential_height"))
        ps_cube = cubes.extract_cube(Constraint(name="surface_air_pressure"))

        # Column-averaged CO2
        xco2_cube = column_average(co2_cube, hus_cube, zg_cube, ps_cube)
        xco2_cube.convert_units("1")

        return xco2_cube
