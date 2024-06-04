"""Fixes for rcm WRF381P driven by IPSL-IPSL-CM5A-MR."""
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


Tasmin = Tas


Tasmax = Tas


Hurs = Tas


Huss = Tas
