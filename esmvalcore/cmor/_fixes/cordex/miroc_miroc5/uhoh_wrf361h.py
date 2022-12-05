"""Fixes for rcm UHOH-WRF361H driven by MIROC-MIROC5."""
import iris
from esmvalcore.cmor.fix import Fix


class Tas(Fix):
    def fix_metadata(self, cubes):
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
