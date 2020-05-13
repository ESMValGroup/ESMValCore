"""Fixes for ACCESS-CM2."""
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
                # values taken from UK-ESM1-0-LL model, which uses the same
                # atmospheric component as ACCESS-CM2 (MetUM-HadGEM3-GA7.1,
                # N96 (192x144), 85 vertical levels, top = 85 km)
                b.points = [
                  0.997741281986237, 0.993982434272766, 0.988731920719147,
                  0.982001721858978, 0.973807096481323, 0.964166879653931,
                  0.953103065490723, 0.940641283988953, 0.926810503005981,
                  0.911642968654633, 0.895174443721771, 0.877444267272949,
                  0.858494758605957, 0.838372051715851, 0.81712543964386,
                  0.7948077917099, 0.77147513628006, 0.747187197208405,
                  0.722006916999817, 0.696000635623932, 0.669238269329071,
                  0.641793012619019, 0.613741397857666, 0.585163474082947,
                  0.556142747402191, 0.526765942573547, 0.49712336063385,
                  0.467308610677719, 0.437418729066849, 0.40755420923233,
                  0.377818822860718, 0.348319888114929, 0.319168090820312,
                  0.290477395057678, 0.262365132570267, 0.234952658414841,
                  0.20836341381073, 0.182725623250008, 0.158169254660606,
                  0.134828746318817, 0.112841464579105, 0.0923482477664948,
                  0.0734933465719223, 0.0564245767891407, 0.041294027119875,
                  0.028257654979825, 0.0174774676561356, 0.00912047084420919,
                  0.00336169824004173, 0.000384818413294852,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            except iris.exceptions.CoordinateNotFoundError:
                pass
        return super().fix_metadata(cubes)


Cli = Cl


Clw = Cl
