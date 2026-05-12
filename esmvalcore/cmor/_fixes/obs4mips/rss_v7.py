"""Fixes for RSS-v7 dataset."""

from esmvalcore.cmor._fixes.fix import Fix


class Prw(Fix):
    """Fixes for prw."""

    def fix_metadata(self, cubes):
        """Fix latitude and longitude varname."""
        for cube in cubes:
            latitude = cube.coord("latitude")
            latitude.var_name = "lat"

            longitude = cube.coord("longitude")
            longitude.var_name = "lon"
        return cubes
