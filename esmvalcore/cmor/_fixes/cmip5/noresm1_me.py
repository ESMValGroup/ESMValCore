"""Fixes for NorESM1-ME model."""
from ..fix import Fix
from ..shared import round_coordinates


class Tas(Fix):
    """Fixes for tas."""

    def fix_metadata(self, cubes):
        """Fix metadata.

        Some coordinate points vary for different files of this dataset (for
        different time range). This fix removes these inaccuracies by rounding
        the coordinates.

        Parameters
        ----------
        cubes: iris.cube.CubeList

        Returns
        -------
        iris.cube.CubeList

        """
        return round_coordinates(cubes, 12)
