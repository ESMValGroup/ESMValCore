"""Fixes for NESM3 model."""
from ..common import ClFixHybridPressureCoord
from ..fix import Fix


Cl = ClFixHybridPressureCoord


Cli = ClFixHybridPressureCoord


Clw = ClFixHybridPressureCoord


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
        bad_units = 'days since 2015-01-01'
        branch_parent = 'branch_time_in_parent'
        bad_value = 1000000
        for cube in cubes:
            try:
                if bad_units in cube.attributes[parent_units] and cube.attributes[branch_parent] > bad_value:
                    cube.attributes[branch_parent] = 0
            except AttributeError:
                pass
        return cubes