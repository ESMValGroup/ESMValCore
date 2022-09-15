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
        self.fix_regular_time(cube)
        self.fix_regular_lat(cube)
        self.fix_regular_lon(cube)

        # Fix scalar coordinates
        self.fix_scalar_coords(cube)

        # Fix metadata of variable
        self.fix_var_metadata(cube)

        return CubeList([cube])
