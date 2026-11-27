"""Derivation of variable ``co2s``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import iris

from esmvalcore.preprocessor._supplementary_vars import add_ancillary_variable
from esmvalcore.preprocessor._volume import extract_surface_from_atm

from ._baseclass import DerivedVariableBase

if TYPE_CHECKING:
    from esmvalcore.typing import Facets


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
    def required(facets: Facets) -> list[Facets]:  # noqa: ARG004
        """Declare the variables needed for derivation."""
        project = facets["project"]
        if project == "CMIP7":
            # For CMIP7, surface CO2 (height=2m) is readily available
            return [
                {"short_name": "co2", "branding_suffix": "tavg-h2m-hxy-u", "mip": "atmos"},
            ]
        else:
            return [{"short_name": "co2"}, {"short_name": "ps"}]

    @staticmethod
    def calculate(cubes):
        """Compute mole fraction of CO2 at surface."""
        try:  # CMIP5, CMIP6
            mrsos = cubes.extract_cube(
                NameConstraint(var_name="mrsos"),
            )
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
        except iris.exceptions.ConstraintMismatchError:
            try:  # CMIP7
                # near-surface CO2 is already available, no need for variable derivation
                # Note: near-surface CO2 from CMIP7 is already in units of 1e-6
                co2s_cube = cubes.extract_cube(
                    iris.Constraint(name="mole_fraction_of_carbon_dioxide_in_air"),
                )
            except iris.exceptions.ConstraintMismatchError as exc:
                msg = (
                    "Derivation of near-surface CO2 failed due to missing "
                    "variable CO2 (co2, ps (CMIP5/CMIP6) or co2 (CMIP7))."
                )
                raise ValueError(msg) from exc
        
        return co2s_cube
