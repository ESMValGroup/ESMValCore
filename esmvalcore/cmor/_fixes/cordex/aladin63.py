"""Fixes for DMI-HIRHAM5 model."""
import iris

from ..common import ClFixHybridPressureCoord
from ..fix import Fix
from ..shared import add_scalar_height_coord

# import IPython
# from traitlets.config import get_config
# c = get_config()
# c.InteractiveShellEmbed.colors = "Linux"
#         IPython.embed(config=c)


from esmvalcore.preprocessor._shared import guess_bounds
import numpy as np

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