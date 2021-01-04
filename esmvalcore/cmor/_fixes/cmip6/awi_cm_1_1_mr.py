"""Fixes for AWI-CM-1-1-MR model."""

from ..fix import Fix


class AllVars(Fix):
    """Fixes for all variables."""

    def fix_metadata(self, cubes):
        """Rename ``long_name`` of latitude to latitude (may be Latitude).

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.Cube
        """
        coords_longnames_to_change = {
            'latitude': 'latitude',
        }

        for cube in cubes:
            for (std_name, long_name) in coords_longnames_to_change.items():
                coord = cube.coord(std_name)
                if coord.long_name != long_name:
                    coord.long_name = long_name

        return cubes
