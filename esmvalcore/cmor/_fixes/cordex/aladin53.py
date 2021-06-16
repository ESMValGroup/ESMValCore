"""Fixes for ALADIN53 model."""
import logging
import iris
import numpy as np

from ..fix import Fix


class AllVars(Fix):
    """Fixes for all variables."""
    def fix_metadata(self, cubes):
        """Fix metadata.

        Fix issue with non monotonicity of dim coords

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList

        """
        for cube in cubes:
            # check dim coords for monotonicity
            for c in ['projection_x_coordinate', 'projection_y_coordinate']:
                points = cube.coord(c).points
                if not iris.util.monotonic(points):
                    # check for regular spacing of (most points)
                    spacings = list(points[1:] - points[:-1])
                    spacing_summary = {
                        x: spacings.count(x)
                        for x in set(spacings)
                    }
                    # sort dict by number of points of each
                    spacing_summary = sorted(spacing_summary.items(),
                                             key=lambda x: x[1],
                                             reverse=True)
                    # check there is only 1 rogue point
                    assert len(spacing_summary) == 3
                    assert spacing_summary[1][1] == 1

                    # recreate points with expected spacing
                    new_points = np.linspace(points[0],
                                             points[-1],
                                             num=len(points))
                    assert new_points[1] - new_points[0] == spacing_summary[0][
                        0]

                    # fix coord
                    cube.coord(c).points = new_points
                    iris.util.promote_aux_coord_to_dim_coord(cube, c)
                    logging.debug('Fixed non monotonic %s', c)

        return cubes


class tas(Fix):
    """Fixes for tas."""
    def fix_metadata(self, cube):
        """
        Fixes incorrect units

        Parameters
        ----------
        cube: iris.cube.Cube
            Input cube.

        Returns
        -------
        iris.cube.Cube

        """
        cube.units = "celsius"
        return cube
