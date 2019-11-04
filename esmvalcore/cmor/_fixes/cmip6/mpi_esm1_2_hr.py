"""Fixes for MPI-ESM1-2-HR model"""

from ..fix import Fix
from ..shared import add_scalar_height_coord


class tas(Fix):
    """ Fixes for tas """

    def fix_metadata(self, cubes):
        """
        Fix metadata.

        Adds missing height2m coordinate.

        Parameters
        ----------
        cube: iris.cube.CubeList

        Returns
        -------
        iris.cube.CubeList

        """
        for cube in cubes:
            add_scalar_height_coord(cube)

        return cubes

class ta(Fix):
    """ Fixes for ta """
    def fix_metadata(self, cubes):
        """
        Fix metadata.

        Corrects plev coordinate var_name.

        Parameters
        ----------
        cube: iris.cube.CubeList

        Returns
        -------
        iris.cube.CubeList

        """
        for cube in cubes:
            plev = cube.coord('air_pressure')
            plev.var_name = 'plev'

        return cubes

class va(Fix):
    """ Fixes for va """
    def fix_metadata(self, cubes):
        """
        Fix metadata.

        Corrects plev coordinate var_name.

        Parameters
        ----------
        cube: iris.cube.CubeList

        Returns
        -------
        iris.cube.CubeList

        """
        for cube in cubes:
            plev = cube.coord('air_pressure')
            plev.var_name = 'plev'

        return cubes

class zg(Fix):
    """ Fixes for zg """
    def fix_metadata(self, cubes):
        """
        Fix metadata.

        Corrects plev coordinate var_name.

        Parameters
        ----------
        cube: iris.cube.CubeList

        Returns
        -------
        iris.cube.CubeList

        """
        for cube in cubes:
            plev = cube.coord('air_pressure')
            plev.var_name = 'plev'

        return cubes

class ua(Fix):
    """ Fixes for ua """
    def fix_metadata(self, cubes):
        """
        Fix metadata.

        Corrects plev coordinate var_name.

        Parameters
        ----------
        cube: iris.cube.CubeList

        Returns
        -------
        iris.cube.CubeList

        """
        for cube in cubes:
            plev = cube.coord('air_pressure')
            plev.var_name = 'plev'

        return cubes

class sfcWind(Fix):
    """ Fixes for sfcWind """

    def fix_metadata(self, cubes):
        """
        Fix metadata.

        Adds missing height10m coordinate.

        Parameters
        ----------
        cube: iris.cube.CubeList

        Returns
        -------
        iris.cube.CubeList

        """
        for cube in cubes:
            add_scalar_height_coord(cube, height=10.0)

        return cubes
