"""Fixes for CMCC-CM2-SR5 model."""
from ..common import ClFixHybridPressureCoord


class Cl(ClFixHybridPressureCoord):
    """Fixes for ``cl``."""

    def fix_metadata(self, cubes):
        """Fix vertical hybrid sigma coordinate (incl. bounds).

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.Cube

        """
        cube = self.get_cube_from_list(cubes)
        ps_coord = cube.coord(var_name='ps')
        ps_coord.standard_name = None
        return super().fix_metadata(cubes)
