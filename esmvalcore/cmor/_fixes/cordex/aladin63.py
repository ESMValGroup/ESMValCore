"""Fixes for CNRM ALADIN63 model."""

from ..fix import Fix
from ..shared import add_scalar_height_coord

class Tas(Fix):
    """Fixes for tas."""

    def fix_metadata(self, cubes):
        """
        Issue: Some file have 10m height for tas
        Add missing height2m coordinate.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList

        """
        for cube in cubes:
            st_names = [coord.standard_name for coord in cube.coords()]
            if 'height' in st_names:
                if cube.coord('height').points[0] != 2.:
                    cube.remove_coord('height')
                    add_scalar_height_coord(cube)
            else:
                add_scalar_height_coord(cube)

        return cubes
