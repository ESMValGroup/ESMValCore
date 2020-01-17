
"""Fix FGOALS-g2 model."""
from cf_units import Unit
from iris.exceptions import CoordinateNotFoundError

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
            try:
                time = cube.coord('time')
            except CoordinateNotFoundError:
                pass
            else:
                time.units = Unit(time.units.name, time.units.calendar)
        return cubes
