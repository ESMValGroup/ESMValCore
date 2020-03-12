"""Common fixes used for multiple datasets."""
import iris

from .fix import Fix
from .shared import fix_bounds


class ClFixHybridPressureCoord(Fix):
    """Fixes for ``cl`` regarding hybrid sigma pressure coordinates."""

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
        cl_cube = self.get_cube_from_list(cubes, short_name='cl')

        # Remove all existing aux_factories
        for aux_factory in cl_cube.aux_factories:
            cl_cube.remove_aux_factory(aux_factory)

        # Fix bounds
        coords_to_fix = ['b']
        try:
            cl_cube.coord(var_name='a')
            coords_to_fix.append('a')
        except iris.exceptions.CoordinateNotFoundError:
            coords_to_fix.append('ap')
        fix_bounds(cl_cube, cubes, coords_to_fix)

        # Fix bounds for ap if only a is given in original file
        # This was originally done by iris, but it has to be repeated since
        # a has bounds now
        ap_coord = cl_cube.coord(var_name='ap')
        if ap_coord.bounds is None:
            cl_cube.remove_coord(ap_coord)
            a_coord = cl_cube.coord(var_name='a')
            p0_coord = cl_cube.coord(var_name='p0')
            ap_coord = a_coord * p0_coord.points[0]
            ap_coord.units = a_coord.units * p0_coord.units
            ap_coord.rename('vertical pressure')
            ap_coord.var_name = 'ap'
            cl_cube.add_aux_coord(ap_coord, cl_cube.coord_dims(a_coord))

        # Add aux_factory again
        pressure_coord_factory = iris.aux_factory.HybridPressureFactory(
            delta=ap_coord,
            sigma=cl_cube.coord(var_name='b'),
            surface_air_pressure=cl_cube.coord(var_name='ps'),
        )
        cl_cube.add_aux_factory(pressure_coord_factory)

        # Remove attributes from Surface Air Pressure coordinate
        cl_cube.coord(var_name='ps').attributes = {}

        return iris.cube.CubeList([cl_cube])
