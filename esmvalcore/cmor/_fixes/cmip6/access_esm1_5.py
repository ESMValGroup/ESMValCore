"""Fixes for ACCESS-ESM1-5."""
import iris

from ..common import ClFixHybridHeightCoord


class Cl(ClFixHybridHeightCoord):
    """Fixes for cl."""

    def fix_metadata(self, cubes):
        """Fix hybrid coefficient b, then call fix_metadata from
           ClFixHybridHeightCoord.

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
                b = cube.coord(var_name='b')
                # values taken from HadGEM2-ES model (CMIP5), which uses the
                # same atmospheric component as ACCESS-ESM1-5 (HadGAM2, N96L38)
                b.points = [
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
                  0, 0, 0, 0, 0, 0, 0, 0, 0]
            except iris.exceptions.CoordinateNotFoundError:
                pass
        return super().fix_metadata(cubes)


Cli = Cl


Clw = Cl
