"""Fixes for rcm WRF381P driven by ICHEC-EC-EARTH."""
from esmvalcore.cmor._fixes.shared import add_scalar_height_coord
from esmvalcore.cmor.fix import Fix


class Tas(Fix):
    """Fixes for tas."""

    def fix_metadata(self, cubes):
        """Add height (2m) coordinate and correct long_name for time.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList
        """
        cube = self.get_cube_from_list(cubes)
        add_scalar_height_coord(cube)

        return cubes


class Tasmin(Tas):
    """Fixes for tasmin."""


class Tasmax(Tas):
    """Fixes for tasmax."""