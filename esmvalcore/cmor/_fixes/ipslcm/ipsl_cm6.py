"""Fixes for IPSLCM6 TS output format."""
import logging
import subprocess
import time

from ..fix import Fix
from ..shared import add_scalar_height_coord

logger = logging.getLogger(__name__)

# The key used in extra_facets file for providing the
# variable name (in NetCDF file) that match the CMOR variable name
VARNAME_KEY = "ipsl_varname"


class AllVars(Fix):
    """Fixes for all IPSLCM variables."""

    def fix_file(self, filepath, output_dir):
        """Select IPSLCM variable in filepath.

        This is done only if input file is a multi-variable one. This
        is diagnosed by searching in the input filepathame for the
        extra_facet value for key 'group'.

        In such cases, it is worth to use an external tool for
        filtering, at least until Iris loads fast (which is not the case
        up to, and including, V3.0.2), and CDO can be used, depending on
        extra_facets key `use_cdo`

        However, we take care of ESMValTool policy re. dependencies licence

        """
        if "_" + self.extra_facets.get("group",
                                       "non-sense") + ".nc" not in filepath:
            # No need to filter the file
            logger.debug("Not filtering for %s", filepath)
            return filepath

        if not self.extra_facets.get("use_cdo", False):
            # The configuration developer doesn't provide CDO, while ESMValTool
            # licence policy doesn't allow to include it in dependencies
            # Or he considers that plain Iris load is quick enough for
            # that file
            logger.debug("In ipsl-cm6.py : CDO not activated for %s", filepath)
            return filepath

        # Proceed with CDO selvar
        varname = self.extra_facets.get(VARNAME_KEY, self.vardef.short_name)
        alt_filepath = filepath.replace(".nc", "_cdo_selected.nc")
        outfile = self.get_fixed_filepath(output_dir, alt_filepath)
        tim1 = time.time()
        logger.debug("Using CDO for selecting %s in %s", varname, filepath)
        command = ["cdo", "-selvar,%s" % varname, filepath, outfile]
        subprocess.run(command, check=True)
        logger.debug("CDO selection done in %.2f seconds", time.time() - tim1)
        return outfile

    def fix_metadata(self, cubes):
        """Fix metadata for any IPSLCM variable + filter out other variables.

        Fix the name of the time coordinate, which is called time_counter
        in the original file.

        Remove standard_name 'time' in auxiliary time coordinates
        """
        logger.debug("Fixing metadata for ipslcm_cm6")

        varname = self.extra_facets.get(VARNAME_KEY, self.vardef.short_name)
        cube = self.get_cube_from_list(cubes, varname)
        cube.var_name = self.vardef.short_name

        # Need to degrade auxiliary time coordinates, because some
        # Iris function does not support to have more than one
        # coordinate with standard_name='time'
        for coordinate in cube.coords(dim_coords=False):
            if coordinate.standard_name == 'time':
                coordinate.standard_name = ''

        # Fix variable name for time_counter
        for coordinate in cube.coords(dim_coords=True):
            if coordinate.var_name == 'time_counter':
                coordinate.var_name = 'time'

        positive = self.extra_facets.get("positive")
        if positive:
            cube.attributes["positive"] = positive

        return [cube]


class Tas(Fix):
    """Fixes for ISPLCM 2m temperature."""

    def fix_metadata(self, cubes):
        """Add height2m."""
        varname = self.extra_facets.get(VARNAME_KEY)
        cube = self.get_cube_from_list(cubes, varname)
        add_scalar_height_coord(cube)
        return cubes


class Huss(Tas):
    """Fixes for ISPLCM 2m specific humidity."""
