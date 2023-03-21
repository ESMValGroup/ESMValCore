"""Fixes for rcm DMI-HIRHAM driven by ICHEC-EC-Earth."""
from esmvalcore.cmor._fixes.shared import add_scalar_height_coord
from esmvalcore.cmor.fix import Fix


class Tas(Fix):
    """Fixes for tas."""

    def fix_metadata(self, cubes):
        """Add height (2m) coordinate.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList
        """
        for cube in cubes:
            add_scalar_height_coord(cube)

        return cubes
