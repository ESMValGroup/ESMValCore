"""Fixes for IPSLCM6 TS output format."""
import logging
import os
import time

from ..fix import Fix
from ..shared import add_scalar_height_coord

logger = logging.getLogger(__name__)

# The key used in mappings.yml file for providing the
# variable name (in NetCDF file) that match the CMOR variable name
KEY_FOR_VARNAME = "ipsl_varname"


class AllVars(Fix):
    """Fixes for all IPSLCM variables."""

    def fix_file(self, filepath, output_dir):
        """Select IPSLCM variable in filepath, by calling CDO, if relevant.

        This is done only if input file is a multi-variable one. This is
        diagnosed by searching in the input filepathame for the mapping
        value for key 'group'.

        In such cases, it is worth to use an external tool for
        filtering, at least until Iris loads fast (which is not the case
        up to, and including, V3.0.2)

        However, we take care of ESMValTool policy re. dependencies licence
        """
        if "_" + self.var_mapping.get("group",
                                      "non-sense") + ".nc" not in filepath:
            # No need to filter the file
            logger.debug("In ipsl-cm6.py : not filtering for %s", filepath)
            return filepath

        if not self.var_mapping.get("use_cdo", False):
            # The configuration developer doesn't provide CDO, while ESMValTool
            # licence policy doesn't allow to include it in dependencies
            # Or he considers that plain Iris load is quick enough for
            # that file
            logger.debug("In ipsl-cm6.py : CDO not activated for %s", filepath)
            return filepath

        # Proceed with CDO selvar
        varname = self.var_mapping.get(KEY_FOR_VARNAME, self.vardef.short_name)
        alt_filepath = filepath.replace(".nc", "_cdo_selected.nc")
        outfile = self.get_fixed_filepath(output_dir, alt_filepath)
        command = "cdo -selvar,%s  %s %s" % (varname, filepath, outfile)
        tim1 = time.time()
        logger.debug("Using CDO for selecting %s in %s", varname, filepath)
        os.system(command)
        logger.debug("CDO selection done in %.2f seconds", time.time() - tim1)
        return outfile

    def fix_metadata(self, cubes):
        """Fix metadata for any IPSLCM variable + filter out other variables.

        Fix the name of the time coordinate, which is called time_counter
        in the original file.

        Remove standard_name 'time' in auxiliary time coordinates
        """
        logger.debug("Fixing metadata for ipslcm_cm6")

        varname = self.var_mapping.get(KEY_FOR_VARNAME, self.vardef.short_name)
        cube = self.get_cube_from_list(cubes, varname)
        cube.var_name = self.vardef.short_name

        # Need to degrade auxiliary time coordinates, because some
        # iris function does not support to have more than one
        # coordinate with standard_name='time'
        for coordinate in cube.coords(dim_coords=False):
            if coordinate.standard_name == 'time':
                coordinate.standard_name = ''

        # Fix variable name for time_counter
        for coordinate in cube.coords(dim_coords=True):
            if coordinate.var_name == 'time_counter':
                coordinate.var_name = 'time'

        return [cube]

    def fix_data(self, cube):
        """Apply fixes to the data of the cube.

        Here : scaling and offset according to mapping.

        But needs to be checked vs ESMValTool automatic unit change
        when units metadat is present and correct
        """
        mapping = self.var_mapping
        metadata = cube.metadata
        if "scale" in mapping:
            cube *= mapping["scale"]
        if "offset" in mapping:
            cube += mapping["offset"]
        cube.metadata = metadata
        return cube


class Tas(Fix):
    """Fixes for ISPLCM 2m temperature."""

    def fix_metadata(self, cubes):
        """Add height2m."""
        varname = self.var_mapping.get(KEY_FOR_VARNAME, self.vardef.short_name)
        cube = self.get_cube_from_list(cubes, varname)
        add_scalar_height_coord(cube)
        return cubes


class Huss(Tas):
    """Fixes for ISPLCM 2m specific humidity."""

    pass
