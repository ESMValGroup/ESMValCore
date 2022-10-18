"""Fixes for rcm MOHC-HadREM3-GA7-05 driven by MOHC-HadGEM2."""
from esmvalcore.cmor.fix import Fix


class Tas(Fix):
    """Fixes for tas."""

    def fix_metadata(self, cubes):
        """
        Fix time long_name.
        Fix latitude and longitude var_name.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList

        """
        for cube in cubes:
          cube.coord('latitude').var_name = 'lat'
          cube.coord('longitude').var_name = 'lon'
          cube.coord('time').long_name = 'time'

        return cubes

Pr = Tas