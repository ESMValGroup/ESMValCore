"""Fixes for IITM-ESM model."""
from ..fix import Fix
from ..shared import fix_ocean_depth_coord



class Omon(Fix):
    """Fixes for ocean variables."""

    def fix_metadata(self, cubes):
        """Fix ocean depth coordinate.
        Parameters
        ----------
        cubes: iris CubeList
            List of cubes to fix
        Returns
        -------
        iris.cube.CubeList
        """
        for cube in cubes:
            if cube.coords('latitude'):
                cube.coord('latitude').var_name = 'lat'
                cube.coord('latitude').guess_bounds()

            if cube.coords('longitude'):
                cube.coord('longitude').var_name = 'lon'
                cube.coord('longitude').guess_bounds()
            if cube.coords(axis='Z'):
                z_coord = cube.coord(axis='Z')
                if z_coord.var_name == 'olevel':
                    fix_ocean_depth_coord(cube)
        return cubes
