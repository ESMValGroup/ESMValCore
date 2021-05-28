"""Fixes for FGOALS-g3 model."""
from ..cmip5.fgoals_g2 import Cl as BaseCl
from ..common import OceanFixGrid

Cl = BaseCl


Cli = BaseCl


Clw = BaseCl


class Tos(OceanFixGrid):
    """Fixes for tos."""

    def fix_metadata(self, cubes):
        """Fix metadata.

        FGOALS-g3 data contain latitude and longitude data set to >1e30 in some
        places.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList

        """
        cube = self.get_cube_from_list(cubes)
        cube.coord('latitude').points[
            cube.coord('latitude').points > 1000.0] = 0.0
        cube.coord('longitude').points[
            cube.coord('longitude').points > 1000.0] = 0.0
        return super().fix_metadata(cubes)


Siconc = Tos
