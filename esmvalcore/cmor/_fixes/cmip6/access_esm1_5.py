"""Fixes for ACCESS-ESM1-5."""
import iris
import numpy as np

from ..common import ClFixHybridHeightCoord
from ..fix import Fix


class Cl(ClFixHybridHeightCoord):
    """Fixes for cl."""

    def fix_metadata(self, cubes):
        """Fix hybrid coefficient b, then call fix_metadata from parent.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList

        """
        for cube in cubes:
            try:
                bcoeff = cube.coord(var_name='b')
                # values taken from HadGEM2-ES model (CMIP5), which uses the
                # same atmospheric component as ACCESS-ESM1-5 (HadGAM2, N96L38)
                bcoeff.points = [
                    0.99771648645401, 0.990881502628326, 0.979542553424835,
                    0.9637770652771, 0.943695485591888, 0.919438362121582,
                    0.891178011894226, 0.859118342399597, 0.823493480682373,
                    0.784570515155792, 0.742646217346191, 0.698050200939178,
                    0.651142716407776, 0.602314412593842, 0.55198872089386,
                    0.500619947910309, 0.44869339466095, 0.39672577381134,
                    0.34526526927948, 0.294891387224197, 0.24621507525444,
                    0.199878215789795, 0.156554222106934, 0.116947874426842,
                    0.0817952379584312, 0.0518637150526047, 0.0279368180781603,
                    0.0107164792716503, 0.00130179093685001,
                    0, 0, 0, 0, 0, 0, 0, 0, 0,
                ]
                bcoeff.bounds = [
                    [1, 0.994296252727509],
                    [0.994296252727509, 0.985203862190247],
                    [0.985203862190247, 0.971644043922424],
                    [0.971644043922424, 0.953709840774536],
                    [0.953709840774536, 0.931527435779572],
                    [0.931527435779572, 0.905253052711487],
                    [0.905253052711487, 0.875074565410614],
                    [0.875074565410614, 0.84121161699295],
                    [0.84121161699295, 0.80391401052475],
                    [0.80391401052475, 0.763464510440826],
                    [0.763464510440826, 0.720175802707672],
                    [0.720175802707672, 0.674392521381378],
                    [0.674392521381378, 0.626490533351898],
                    [0.626490533351898, 0.576877355575562],
                    [0.576877355575562, 0.525990784168243],
                    [0.525990784168243, 0.474301367998123],
                    [0.474301367998123, 0.422309905290604],
                    [0.422309905290604, 0.370548874139786],
                    [0.370548874139786, 0.3195820748806],
                    [0.3195820748806, 0.270004868507385],
                    [0.270004868507385, 0.222443267703056],
                    [0.222443267703056, 0.177555426955223],
                    [0.177555426955223, 0.136030226945877],
                    [0.136030226945877, 0.0985881090164185],
                    [0.0985881090164185, 0.0659807845950127],
                    [0.0659807845950127, 0.0389823913574219],
                    [0.0389823913574219, 0.0183146875351667],
                    [0.0183146875351667, 0.00487210927531123],
                    [0.00487210927531123, 0],
                    [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
                    [0, 0], [0, 0],
                ]
            except iris.exceptions.CoordinateNotFoundError:
                pass
        return super().fix_metadata(cubes)


Cli = Cl


Clw = Cl


class Hus(Fix):
    """Fixes for hus."""

    def fix_metadata(self, cubes):
        """Correctly round air pressure coordinate.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.Cube

        """
        cube = self.get_cube_from_list(cubes)
        cube.coord('air_pressure').points = \
            np.round(cube.coord('air_pressure').points, 0)
        cube.coord('air_pressure').bounds = \
            np.round(cube.coord('air_pressure').bounds, 0)
        return cubes


class Zg(Fix):
    """Fixes for zg."""

    def fix_metadata(self, cubes):
        """Correctly round air pressure coordinate.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.Cube

        """
        cube = self.get_cube_from_list(cubes)
        cube.coord('air_pressure').points = \
            np.round(cube.coord('air_pressure').points, 0)
        cube.coord('air_pressure').bounds = \
            np.round(cube.coord('air_pressure').bounds, 0)
        return cubes
