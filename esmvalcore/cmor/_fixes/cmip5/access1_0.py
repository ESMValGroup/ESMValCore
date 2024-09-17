"""Fixes for the ACCESS1-0 model."""
import iris
from cf_units import Unit

from ..common import ClFixHybridHeightCoord
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
        for cube in cubes:
            try:
                time = cube.coord('time')
            except iris.exceptions.CoordinateNotFoundError:
                continue
            else:
                if time.units.calendar == 'proleptic_gregorian':
                    time.convert_units(Unit("days since 1850-01-01",
                                            calendar='proleptic_gregorian'))
                    time.units = Unit(time.units.name, 'gregorian')
        return cubes


class Cl(ClFixHybridHeightCoord):
    """Fixes for ``cl``."""

    def fix_metadata(self, cubes):
        """Remove attributes from ``vertical coordinate formula term: b(k)``.

        Additionally add pressure level coordiante.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes which need to be fixed.

        Returns
        -------
        iris.cube.CubeList

        """
        cubes = super().fix_metadata(cubes)
        cube = self.get_cube_from_list(cubes)
        coord = cube.coord(long_name='vertical coordinate formula term: b(k)')
        coord.attributes = {}
        return cubes
