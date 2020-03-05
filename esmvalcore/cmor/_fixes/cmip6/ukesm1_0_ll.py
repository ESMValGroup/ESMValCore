"""Fixes for CMIP6 UKESM1-0-LL."""
import iris

from ..fix import Fix
from ..shared import add_pressure_level_coordinate, fix_bounds
from .hadgem3_gc31_ll import AllVars as BaseAllVars


class AllVars(BaseAllVars):
    """Fixes for all vars."""


class Cl(Fix):
    """Fixes for ``'cl'``."""

    def fix_metadata(self, cubes):
        """Fix hybrid sigma height coordinate and add pressure levels.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes which need to be fixed.

        Returns
        -------
        iris.cube.CubeList

        """
        cl_cube = self.get_cube_from_list(cubes)

        # Remove all existing aux_factories
        for aux_factory in cl_cube.aux_factories:
            cl_cube.remove_aux_factory(aux_factory)

        # Fix bounds
        fix_bounds(cl_cube, cubes, ('lev', 'b'))

        # Add aux_factory again
        height_coord_factory = iris.aux_factory.HybridHeightFactory(
            delta=cl_cube.coord(var_name='lev'),
            sigma=cl_cube.coord(var_name='b'),
            orography=cl_cube.coord(var_name='orog'),
        )
        cl_cube.add_aux_factory(height_coord_factory)

        # Add pressure level coordinate
        add_pressure_level_coordinate(cl_cube)

        return iris.cube.CubeList([cl_cube])


class Clw(Cl):
    """Fixes for ``clw (same as for cl)``."""


class Cli(Cl):
    """Fixes for ``cli (same as for cl)``."""
