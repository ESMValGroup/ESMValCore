"""Fixes for rcm HIRHAM driven by MOHC-HadGEM2."""
from esmvalcore.cmor.fix import Fix


class Pr(Fix):
    """Fixes for pr."""

    def fix_metadata(self, cubes):
        """Remove latitude and longitude attributes.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList

        """
        for cube in cubes:
            cube.coord('latitude').attributes = {}
            cube.coord('longitude').attributes = {}

        return cubes
