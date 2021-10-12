"""Fix UKCP land-rcm data."""
from ..fix import Fix


class AllVars(Fix):
    """Fix errors common to all vars."""
    def fix_metadata(self, cubes):
        """Fix latitude and longitude.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes which need to be fixed.

        Returns
        -------
        iris.cube.CubeList

        """
        for cube in cubes:
            lats = cube.coords("latitude")
            if lats:
                lat = cube.coord("latitude")
                lat.var_name = "lat"
                lat.coord_system = None

            lons = cube.coords("longitude")
            if lons:
                lon = cube.coord("longitude")
                lon.var_name = "lon"
                lon.coord_system = None

        return cubes
