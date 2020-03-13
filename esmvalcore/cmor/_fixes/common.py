"""Common fixes used for multiple datasets."""
import iris

from .fix import Fix
from .shared import add_plev_from_altitude, fix_bounds


class ClFixHybridHeightCoord(Fix):
    """Fixes for ``cl`` regarding hybrid sigma height coordinates."""

    SHORT_NAME = 'cl'

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
        cube = self.get_cube_from_list(cubes, short_name=self.SHORT_NAME)

        # Remove all existing aux_factories
        for aux_factory in cube.aux_factories:
            cube.remove_aux_factory(aux_factory)

        # Fix bounds
        fix_bounds(cube, cubes, ('lev', 'b'))

        # Add aux_factory again
        height_coord_factory = iris.aux_factory.HybridHeightFactory(
            delta=cube.coord(var_name='lev'),
            sigma=cube.coord(var_name='b'),
            orography=cube.coord(var_name='orog'),
        )
        cube.add_aux_factory(height_coord_factory)

        # Add pressure level coordinate
        add_plev_from_altitude(cube)

        return iris.cube.CubeList([cube])


class CliFixHybridHeightCoord(ClFixHybridHeightCoord):
    """Fixes for ``cli`` regarding hybrid sigma height coordinates."""

    SHORT_NAME = 'cli'


class ClwFixHybridHeightCoord(ClFixHybridHeightCoord):
    """Fixes for ``clw`` regarding hybrid sigma height coordinates."""

    SHORT_NAME = 'clw'


class ClFixHybridPressureCoord(Fix):
    """Fixes for ``cl`` regarding hybrid sigma pressure coordinates."""

    SHORT_NAME = 'cl'

    def fix_metadata(self, cubes):
        """Fix hybrid sigma pressure coordinate.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes which need to be fixed.

        Returns
        -------
        iris.cube.CubeList

        """
        cube = self.get_cube_from_list(cubes, short_name=self.SHORT_NAME)

        # Remove all existing aux_factories
        for aux_factory in cube.aux_factories:
            cube.remove_aux_factory(aux_factory)

        # Fix bounds
        coords_to_fix = ['b']
        try:
            cube.coord(var_name='a')
            coords_to_fix.append('a')
        except iris.exceptions.CoordinateNotFoundError:
            coords_to_fix.append('ap')
        fix_bounds(cube, cubes, coords_to_fix)

        # Fix bounds for ap if only a is given in original file
        # This was originally done by iris, but it has to be repeated since
        # a has bounds now
        ap_coord = cube.coord(var_name='ap')
        if ap_coord.bounds is None:
            cube.remove_coord(ap_coord)
            a_coord = cube.coord(var_name='a')
            p0_coord = cube.coord(var_name='p0')
            ap_coord = a_coord * p0_coord.points[0]
            ap_coord.units = a_coord.units * p0_coord.units
            ap_coord.rename('vertical pressure')
            ap_coord.var_name = 'ap'
            cube.add_aux_coord(ap_coord, cube.coord_dims(a_coord))

        # Add aux_factory again
        pressure_coord_factory = iris.aux_factory.HybridPressureFactory(
            delta=ap_coord,
            sigma=cube.coord(var_name='b'),
            surface_air_pressure=cube.coord(var_name='ps'),
        )
        cube.add_aux_factory(pressure_coord_factory)

        # Remove attributes from Surface Air Pressure coordinate
        cube.coord(var_name='ps').attributes = {}

        return iris.cube.CubeList([cube])


class CliFixHybridPressureCoord(ClFixHybridPressureCoord):
    """Fixes for ``cli`` regarding hybrid sigma pressure coordinates."""

    SHORT_NAME = 'cli'


class ClwFixHybridPressureCoord(ClFixHybridPressureCoord):
    """Fixes for ``clw`` regarding hybrid sigma pressure coordinates."""

    SHORT_NAME = 'clw'
