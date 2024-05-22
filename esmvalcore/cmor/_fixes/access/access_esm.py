"""On-the-fly CMORizer for ACCESS-ESM.

Note
----
This is the first version of ACCESS-ESM CMORizer in for ESMValCore
Currently, only two variables (`tas`,`pr`) is fully supported.
"""
import logging
import os

from iris.cube import CubeList

from esmvalcore.cmor.check import cmor_check

from ..native_datasets import NativeDatasetFix

logger = logging.getLogger(__name__)


class Tas(NativeDatasetFix):
    """Fix variable(tas) only."""

    def __init__(self, vardef, extra_facets, session, frequency):
        """Initialise some class variable Heritage from native_dataset."""
        super().__init__(vardef, extra_facets, session, frequency)

        self.cube = None

        self.current_dir = os.path.dirname(__file__)

    def _fix_height_name(self):
        """Fix variable name of coordinate 'height'."""
        if self.cube.coord('height').var_name != 'height':
            self.cube.coord('height').var_name = 'height'

    def _fix_long_name(self):
        """Fix variable long_name."""
        self.cube.long_name = 'Near-Surface Air Temperature'

    def _fix_var_name(self):
        """Fix variable long_name."""
        self.cube.var_name = 'tas'

    def fix_coord_system(self):
        """Delete coord_system to make CubeList able to merge."""
        for dim in self.cube.dim_coords:
            if dim.coord_system is not None:
                self.cube.coord(dim.standard_name).coord_system = None

    def fix_metadata(self, cubes):
        """Fix metadata.

        Fix name of coordinate(height), long name and variable name of
        variable(tas).

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList
        """
        original_short_name = 'fld_s03i236'

        self.cube = self.get_cube(cubes, var_name=original_short_name)

        self.fix_height_name()

        self.fix_long_name()

        self.fix_var_name()

        self.fix_coord_system()

        cube_checked = cmor_check(cube=self.cube,
                                  cmor_table='CMIP6',
                                  mip='Amon',
                                  short_name='tas',
                                  check_level=1)

        return CubeList([cube_checked])


class Pr(NativeDatasetFix):
    """Fix variable(pr) only."""

    def __init__(self, vardef, extra_facets, session, frequency):
        """Initialise some class variable Heritage from native_dataset."""
        super().__init__(vardef, extra_facets, session, frequency)

        self.cube = None

        self.current_dir = os.path.dirname(__file__)

    def fix_var_name(self):
        """Fix variable long_name."""
        self.cube.var_name = 'pr'

    def fix_long_name(self):
        """Fix variable long_name."""
        self.cube.long_name = 'Precipitation'

    def fix_coord_system(self):
        """Delete coord_system to make CubeList able to merge."""
        for dim in self.cube.dim_coords:
            if dim.coord_system is not None:
                self.cube.coord(dim.standard_name).coord_system = None

    def fix_metadata(self, cubes):
        """Fix metadata.

        Fix name of coordinate(height), long name and variable name of
        variable(tas).

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList
        """
        original_short_name = 'fld_s05i216'

        cube = self.get_cube(cubes, var_name=original_short_name)

        self.fix_var_name()

        self.fix_long_name()

        self.fix_coord_system()

        cube_checked = cmor_check(cube=cube,
                                  cmor_table='CMIP6',
                                  mip='Amon',
                                  short_name='pr',
                                  check_level=1)

        return CubeList([cube_checked])
