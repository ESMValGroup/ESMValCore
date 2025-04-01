"""Derivation of variable `qep`."""

from iris import Constraint

from ._baseclass import DerivedVariableBase


class DerivedVariable(DerivedVariableBase):
    """Derivation of variable `qep`."""

    @staticmethod
    def required(project):
        """Declare the variables needed for derivation."""
        required = [
            {"short_name": "evspsbl"},
            {"short_name": "pr"},
        ]
        return required

    @staticmethod
    def calculate(cubes):
        """Compute net moisture flux into atmosphere."""
        evspsbl_cube = cubes.extract_cube(
            Constraint(name="water_evapotranspiration_flux")
        )
        pr_cube = cubes.extract_cube(
            Constraint(name="precipitation_flux")
        )

        qep_cube = evspsbl_cube - pr_cube
        qep_cube.units = pr_cube.units
        qep_cube.attributes["positive"] = "up"

        return qep_cube
