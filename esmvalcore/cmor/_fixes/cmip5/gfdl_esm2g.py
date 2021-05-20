
"""Fixes for GFDL ESM2G."""

import iris

from ..fix import Fix


def _get_and_remove(cubes, long_name):
    try:
        cube = cubes.extract_cube(long_name)
        cubes.remove(cube)
    except iris.exceptions.ConstraintMismatchError:
        pass


class AllVars(Fix):
    """Common fixes."""

    def fix_metadata(self, cubes):
        """Fix metadata.

        Fixes bad standard names.

        Parameters
        ----------
        cubes: iris.cube.CubeList

        Returns
        -------
        iris.cube.CubeList

        """
        _get_and_remove(cubes, 'Start time for average period')
        _get_and_remove(cubes, 'End time for average period')
        _get_and_remove(cubes, 'Length of average period')
        return cubes


class Areacello(Fix):
    """Fixes for areacello"""

    def fix_metadata(self, cubes):
        """
        Fix metadata.

        Fixes wrong units.

        Parameters
        ----------
        cube: iris.cube.Cube

        Returns
        -------
        iris.cube.Cube

        """
        cube = self.get_cube_from_list(cubes)
        cube.units = 'm2'
        return cubes


class Co2(Fix):
    """Fixes for co2."""

    def fix_data(self, cube):
        """Fix data.

        Fixes discrepancy between declared units and real units.

        Parameters
        ----------
        cube: iris.cube.Cube

        Returns
        -------
        iris.cube.Cube

        """
        metadata = cube.metadata
        cube *= 1e6
        cube.metadata = metadata
        return cube


class FgCo2(Fix):
    """Fixes for fgco2."""

    def fix_metadata(self, cubes):
        """Fix metadata.

        Remove unnecessary variables prohibiting cube concatenation.

        Parameters
        ----------
        cubes: iris.cube.CubeList

        Returns
        -------
        iris.cube.CubeList

        """
        _get_and_remove(cubes, 'Latitude of tracer (h) points')
        _get_and_remove(cubes, 'Longitude of tracer (h) points')
        return cubes


class Usi(Fix):
    """Fixes for usi."""

    def fix_metadata(self, cubes):
        """
        Fix metadata.

        Fixes bad standard_name

        Parameters
        ----------
        cubes: iris.cube.CubeList
        Returns
        -------
        iris.cube.CubeList
        """
        cube = self.get_cube_from_list(cubes)
        cube.standard_name = 'sea_ice_x_velocity'
        return cubes


class Vsi(Fix):
    """Fixes for vsi."""

    def fix_metadata(self, cubes):
        """
        Fix metadata.

        Fixes bad standard_name

        Parameters
        ----------
        cubes: iris.cube.CubeList
        Returns
        -------
        iris.cube.CubeList
        """
        cube = self.get_cube_from_list(cubes)
        cube.standard_name = 'sea_ice_y_velocity'
        return cubes
