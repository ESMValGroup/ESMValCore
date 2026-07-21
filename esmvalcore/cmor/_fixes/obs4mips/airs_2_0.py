"""Fixes for obs4MIPs dataset AIRS-2-0."""

from esmvalcore.cmor._fixes.fix import Fix


class Hur(Fix):
    """Fixes for hur."""

    def fix_metadata(self, cubes):
        """Fix metadata.

        Convert units from `1` to `%`.

        Parameters
        ----------
        cubes: iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList
            Fixed cubes.

        """
        for cube in cubes:
            cube.convert_units("%")
        return cubes
