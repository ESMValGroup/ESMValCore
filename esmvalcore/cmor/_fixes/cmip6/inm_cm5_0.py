"""Fixes for CMIP6 INM-CM5-0."""
from ..fix import Fix

class Nbp(Fix):
    """Fixes for fgco2."""

    def fix_data(self, cube):
        """
        Fix data.

        Reported in kg of CO2 rather than kg of carbon.

        Parameters
        ----------
        cube: iris.cube.Cube

        Returns
        -------
        iris.cube.Cube

        """
        metadata = cube.metadata
        cube *= 12./44.
        cube.metadata = metadata
        return cube
