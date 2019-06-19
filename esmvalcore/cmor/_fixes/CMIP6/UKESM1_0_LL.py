# pylint: disable=invalid-name, no-self-use, too-few-public-methods
"""Fixes for UKESM1-0-LL model."""
from ..fix import Fix


class mrsos(Fix):
    """Fixes for mrsos."""

    def fix_metadata(self, cubes):
        """
        Fix metadata.

        Fixes wrong standard name.
        Original error: mrsos: standard_name should be
        moisture_content_of_soil_layer,
        not mass_content_of_water_in_soil_layer.

        Parameters
        ----------
        cube: iris.cube.Cube

        Returns
        -------
        iris.cube.Cube

        """
        for cube in cubes:
            cube.standard_name = 'moisture_content_of_soil_layer'
        return cubes
