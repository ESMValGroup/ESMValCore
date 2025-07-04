"""Derivation of variable `rsnstcs`.

authors:
    - weig_ka

"""

from iris import Constraint

from ._baseclass import DerivedVariableBase


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `rsnstcs`."""

    @staticmethod
    def required(project):
        """Declare the variables needed for derivation."""
        return [
            {"short_name": "rsdscs"},
            {"short_name": "rsdt"},
            {"short_name": "rsuscs"},
            {"short_name": "rsutcs"},
        ]

    @staticmethod
    def calculate(cubes):
        """Compute Heating from Shortwave Absorption assuming clear sky."""
        rsdscs_cube = cubes.extract_cube(
            Constraint(
                name=(
                    "surface_downwelling_shortwave_flux_in_air_"
                    "assuming_clear_sky"
                ),
            ),
        )
        rsdt_cube = cubes.extract_cube(
            Constraint(name="toa_incoming_shortwave_flux"),
        )
        rsuscs_cube = cubes.extract_cube(
            Constraint(
                name=(
                    "surface_upwelling_shortwave_flux_in_air_"
                    "assuming_clear_sky"
                ),
            ),
        )
        rsutcs_cube = cubes.extract_cube(
            Constraint(name="toa_outgoing_shortwave_flux_assuming_clear_sky"),
        )

        return (rsdt_cube - rsutcs_cube) - (rsdscs_cube - rsuscs_cube)
