"""Fixes for MCM-UA-1-0 model."""
import iris

from ..fix import Fix


class AllVars(Fix):
    """Fixes for all variables."""

    def fix_metadata(self, cubes):
        """Fix metadata.

        Remove unnecessary data contact attribute

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Cubes to fix.

        Returns
        -------
        iris.cube.Cube

        """
        for cube in cubes:
            try:
                if cube.attributes['data contact']:
                    del cube.attributes['data contact']
            except KeyError:
                pass
        return cubes
