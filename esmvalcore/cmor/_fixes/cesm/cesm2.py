"""On-the-fly CMORizer for CESM2.

Warning
-------
The support for native CESM output is still experimental. Currently, only one
variable (`tas`) is fully supported. Other 2D variables might be supported by
specifying appropriate facets in the recipe or extra facets files (see
doc/quickstart/find_data.rst for details). 3D variables are currently not
supported.

To add support for more variables, expand the extra facets file
(esmvalcore/_config/extra_facets/cesm-mappings.yml) and/or add classes to this
file for variables that need more complex fixes (see
esmvalcore/cmor/_fixes/emac/emac.py for examples).

"""

import logging

from iris.cube import CubeList

from ..native_datasets import NativeDatasetFix

logger = logging.getLogger(__name__)


class AllVars(NativeDatasetFix):
    """Fixes for all variables."""

    # Dictionary to map invalid units in the data to valid entries
    INVALID_UNITS = {
        'fraction': '1',
    }

    def fix_metadata(self, cubes):
        """Fix metadata."""
        cube = self.get_cube(cubes)

        # Fix time, latitude, and longitude coordinates
        # Note: 3D variables are currently not supported
        self._fix_time(cube)
        self.fix_regular_lat(cube)
        self.fix_regular_lon(cube)

        # Fix scalar coordinates
        self.fix_scalar_coords(cube)

        # Fix metadata of variable
        self.fix_var_metadata(cube)

        return CubeList([cube])

    def _fix_time(self, cube):
        """Fix time coordinate of cube.

        For monthly data that does not correspond to point measurements, move
        time points to center of time period given by time bounds (currently
        the points are located at the end of the interval).

        Example of monthly time coordinate before this fix (Jan. & Feb. 2000):
            Points: ``[2000-02-01, 2000-03-01]``
            Bounds: ``[[2000-01-01, 2000-02-01], [2000-02-01, 2000-03-01]]``

        Example of monthly time coordinate after this fix (Jan. & Feb. 2000):
            Points: ``[2000-01-15, 2000-02-14]``
            Bounds: ``[[2000-01-01, 2000-02-01], [2000-02-01, 2000-03-01]]``

        """
        # Only modify time points if data contains a time dimension, is monthly
        # data, and does not describe point measurements.
        if not self.vardef.has_coord_with_standard_name('time'):
            return
        if self.extra_facets['frequency'] != 'mon':
            return
        for cell_method in cube.cell_methods:
            if 'point' in cell_method.method:
                return

        # Fix time coordinate
        time_coord = cube.coord('time')
        if time_coord.has_bounds():
            time_coord.points = time_coord.core_bounds().mean(axis=-1)
        self.fix_regular_time(cube, coord=time_coord)
