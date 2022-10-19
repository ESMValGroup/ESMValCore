"""Fixes for rcm GERICS-REMO2015 driven by NCC-NorESM1-M."""
from esmvalcore.cmor.fix import Fix


class Pr(Fix):
    """Fixes for pr."""

    def fix_metadata(self, cubes):
        """
        Fix time long_name.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList

        """
        for cube in cubes:
            cube.coord('time').long_name = 'time'            

        return cubes
