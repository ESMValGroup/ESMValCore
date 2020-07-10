"""Fixes for the ACCESS1-0 model."""
from esmvalcore.preprocessor._shared import guess_bounds

from ..fix import Fix

class AllVars(Fix):
    """Common fixes to all vars."""

    def fix_metadata(self, cubes):
        """
        Fix metadata.

        Fixes wrong calendar 'gregorian' instead of 'proleptic_gregorian'.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes which need to be fixed.

        Returns
        -------
        iris.cube.CubeList

        """
        # import IPython
        # from traitlets.config import get_config
        # c = get_config()
        # c.InteractiveShellEmbed.colors = "Linux"

        # IPython.embed(config=c)

        for cube in cubes:
            cube = guess_bounds(cube, ['time'])
        return cubes
