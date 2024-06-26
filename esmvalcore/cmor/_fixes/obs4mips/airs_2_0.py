"""Fixes for obs4MIPs dataset AIRS-2-0."""
import dask.array as da

from ..fix import Fix


class Hur(Fix):
    """Fixes for hur."""

    def fix_metadata(self, cubes):
        """Fix metadata.

        Convert units from `1` to `%` and remove `valid_range` attribute.

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
            # Put information from valid_range into mask and remove the
            # attribute (otherwise this will cause problems after reloading the
            # data with different units)
            valid_range = cube.attributes['valid_range']
            cube.data = da.ma.masked_outside(cube.core_data(), *valid_range)
            cube.attributes.pop('valid_range', None)

            cube.convert_units('%')
        return cubes
