"""Fixes for ICON-ESM-LR model."""

from ..fix import Fix


class AllVars(Fix):
    """Fixes for all variables."""

    def fix_metadata(self, cubes):
        """Rename ``var_name`` of latitude and longitude.

        Parameters
        ----------
        cubes: iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList
            Fixed cubes.

        """
        varnames_to_change = {
            'latitude': 'lat',
            'longitude': 'lon',
        }

        for cube in cubes:
            for (std_name, var_name) in varnames_to_change.items():
                if cube.coords(std_name):
                    cube.coord(std_name).var_name = var_name

        return cubes
