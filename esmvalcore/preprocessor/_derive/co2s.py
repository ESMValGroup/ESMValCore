"""Derivation of variable ``co2s``."""

import iris

from esmvalcore.preprocessor._supplementary_vars import add_ancillary_variable
from esmvalcore.preprocessor._volume import extract_surface_from_atm

from ._baseclass import DerivedVariableBase


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable ``co2s``.

    Use linear interpolation/extrapolation and surface air pressure to
    calculate CO2 mole fraction at surface.

    Note
    ----
    In some cases, ``co2`` data is masked. In these cases, the masked values
    correspond to values where the pressure level is higher than the surface
    air pressure (e.g. the 1000 hPa level for grid cells with high elevation).
    To obtain an unmasked ``co2s`` field, it is necessary to fill these masked
    values accordingly, i.e. with the lowest unmasked value for each grid cell.

    """

    @staticmethod
    def required(project):  # noqa: ARG004
        """Declare the variables needed for derivation."""
        return [{"short_name": "co2"}, {"short_name": "ps"}]

    @staticmethod
    def calculate(cubes):
        """Compute mole fraction of CO2 at surface."""
        co2_cube = cubes.extract_cube(
            iris.Constraint(name="mole_fraction_of_carbon_dioxide_in_air"),
        )
        ps_cube = cubes.extract_cube(
            iris.Constraint(name="surface_air_pressure"),
        )
        # Add ps as AncillaryVariable in the CO2 cube
        add_ancillary_variable(cube=co2_cube, ancillary_cube=ps_cube)
        # Extract surface from 3D atmospheric cube
        co2s_cube = extract_surface_from_atm(co2_cube)
        co2s_cube.convert_units("1e-6")
        return co2s_cube
