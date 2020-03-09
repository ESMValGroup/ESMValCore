"""Fixes for CMIP6 UKESM1-0-LL."""
import iris

from ..fix import Fix
from ..shared import add_plev_from_altitude, fix_bounds


class AllVars(Fix):
    """Fixes for all vars."""

    def fix_metadata(self, cubes):
        """Fix parent time units.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList

        """
        parent_units = 'parent_time_units'
        bad_value = 'days since 1850-01-01-00-00-00'
        for cube in cubes:
            try:
                if cube.attributes[parent_units] == bad_value:
                    cube.attributes[parent_units] = 'days since 1850-01-01'
            except AttributeError:
                pass
        return cubes


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
        add_plev_from_altitude(cl_cube)

        return iris.cube.CubeList([cl_cube])


class Clw(Cl):
    """Fixes for ``clw``."""


class Cli(Cl):
    """Fixes for ``cli``."""
