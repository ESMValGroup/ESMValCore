
"""Fix FGOALS-g2 model."""
from cf_units import Unit
from iris.exceptions import CoordinateNotFoundError

from ..fix import Fix
from ..shared import round_coordinates


class AllVars(Fix):
    """Fix errors common to all vars."""

    def fix_metadata(self, cubes):
        """
        Fix metadata.

        Fixes time units

        Fixes longitude precision

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

        round_coordinates(cubes, 4, coord_names=['longitude'])

        return cubes
