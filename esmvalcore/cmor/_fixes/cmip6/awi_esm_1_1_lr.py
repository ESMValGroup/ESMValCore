"""Fixes for AWI-ESM-1-1-LR model."""
from ..fix import Fix


class AllVars(Fix):
    """Fixes for all vars."""

    def fix_metadata(self, cubes):
        """Fix parent time units.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList
        """
        parent_units = 'parent_time_units'
        bad_value = 'days since 0000-01-01 00:00:00'
        for cube in cubes:
            try:
                if cube.attributes[parent_units] == bad_value:
                    cube.attributes[parent_units] = 'days since 0001-01-01 ' \
                        + '00:00:00'
            except AttributeError:
                pass
        return cubes
