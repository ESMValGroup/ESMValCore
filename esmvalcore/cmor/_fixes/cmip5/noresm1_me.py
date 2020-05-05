"""Fixes for NorESM1-ME model."""
from ..fix import Fix
from ..shared import round_coordinates


class Pr(Fix):
    """Fix errors for pr."""

    def fix_metadata(self, cubes):
        """Fix metadata.

        Fixes latitude coordinate inaccuracies

        Data is fixed in-place, as well as returned

        Parameters
        ----------
        cube: iris.cube.CubeList

        Returns
        -------
        iris.cube.CubeList

        """
        return round_coordinates(cubes, 12, coord_names=['latitude'])


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
