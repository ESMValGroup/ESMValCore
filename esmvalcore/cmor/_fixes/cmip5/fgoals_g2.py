
"""Fix FGOALS-g2 model."""
from cf_units import Unit

from ..fix import Fix


class AllVars(Fix):
    """Fix errors common to all vars."""

    def fix_metadata(self, cubes):
        """
        Fix metadata.

        Fixes time units

        Parameters
        ----------
        cube: iris.cube.CubeList

        Returns
        -------
        iris.cube.Cube

        """
        for cube in cubes:
            time = cube.coord('time')
            time.units = Unit(time.units.name, time.units.calendar)
        return cubes
