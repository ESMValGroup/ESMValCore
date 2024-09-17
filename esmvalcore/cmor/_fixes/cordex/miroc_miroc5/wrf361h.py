"""Fixes for rcm WRF361H driven by MIROC-MIROC5."""
import iris
from esmvalcore.cmor.fix import Fix


class Tas(Fix):
    """Fixes for tas."""

    def fix_metadata(self, cubes):
        """Fix tas coordinates.

        Set height as an auxiliary coordinate instead
        of as a dimensional coordinate.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList

        """
        fixed_cubes = iris.cube.CubeList()
        for cube in cubes:
            height = cube.coord('height')
            if isinstance(height, iris.coords.DimCoord):
                iris.util.demote_dim_coord_to_aux_coord(
                    cube,
                    height
                )
            fixed_cubes.append(iris.util.squeeze(cube))
        return fixed_cubes
