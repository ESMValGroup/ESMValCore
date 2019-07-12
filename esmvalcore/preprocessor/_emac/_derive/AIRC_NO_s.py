"""Derivation of variable `AIRC_NO_s`."""
"""The variable 'AIRC_NO_s' is an EMAC variable that is used """
"""for monitoring EMAC output. It is here integrated over all """
"""available levels (with the help of the field 'geopot_ave'). """
"""The variable is stored in the EMAC CMIP6 channel 'import_grid'. """
"""AIRC_NO_s: Aircraft NO, summed """

import logging

import iris
import iris.analysis
import iris.coord_categorisation
from . import var_name_constraint
import cf_units
from scipy import constants

logger = logging.getLogger(__name__)


def derive(cubes):
    """Calculate vertical integral of aircraft emissions (on model levels):
     emissions are given in kg m^-3 s^-1 and must be multiplied with the layer
     thickness (m) before summing up all vertical levels.
     Note:
     (1) The aircraft emissions and the geopotential height fields are
         from different EMAC channel and thus may contain a different number
         of time steps. In that case, we calculate monthly means and try
         again. If the number of time steps is still different, we have to
         surrender and throw an error message.
     (2) The emissions contain an invalid unit (kg(NO2) m^-3 s^-1) so
         we overwrite the unit with kg m^-3 s^-1.
    """
    no_cube = cubes.extract_strict(var_name_constraint('airc_NO'))
    geopot_cube = cubes.extract_strict(var_name_constraint('geopot_ave'))

    time_no = no_cube.coord('time')
    time_geopot = geopot_cube.coord('time')

    # overwrite unit (kg(NO2) m^-3 s-^ ---> kg m^-3 s^-1)
    no_cube.units = "kg m^-3 s^-1"

    # check if number of time steps in both fields (emissions and geopotential)
    # is equal; if not, calculate monthly means of both fields
    if (time_no.shape != time_geopot.shape):
        iris.coord_categorisation.add_month(no_cube, 'time', name='month')
        iris.coord_categorisation.add_month(geopot_cube, 'time', name='month')
        iris.coord_categorisation.add_year(no_cube, 'time', name='year')
        iris.coord_categorisation.add_year(geopot_cube, 'time', name='year')
        no_cube = no_cube.aggregated_by(['month', 'year'], iris.analysis.MEAN)
        geopot_cube = geopot_cube.aggregated_by(['month', 'year'],
                                                iris.analysis.MEAN)

        if (no_cube.shape != geopot_cube.shape):
            # the number of time steps in both fields still differs
            # --> now we have to surrender...
            logger.error(
                "Could not calculate vertical integral of EMAC aircraft " +
                "emissions. Abort.")
            raise Exception("inconsistent number of time steps")

    # calculate layer thickness (m)
    coords = no_cube.coords()
    z_coord = coords[1]
    z_coord_name = z_coord.name()
    layer_widths = _level_widths(geopot_cube, z_coord)

    # multiply aircraft emissions with layer thickness and calculate sum
    # over all levels
    no_s_cube = no_cube * layer_widths
    no_s_cube = no_s_cube.collapsed(z_coord_name, iris.analysis.SUM)
    no_s_cube.units = (no_cube.units * layer_widths.units)

    return (no_s_cube)


# Helper functions
def _level_widths(geopot_cube, z_coord):
    """Create a cube with height level widths.
    Parameters
    ----------
        geopot_cube: geopotential height (m^2 s^-2)
        z_coord: vertical coordinate (model levels)
    Returns
    -------
    iris.cube.Cube
    `Cube` of same shape as `geopot_cube` containing height level widths (m).
    """
    # Constants
    STANDARD_GRAVITY = 9.81
    STANDARD_GRAVITY_UNIT = cf_units.Unit('m s^-2')

    delta_z_cube = geopot_cube.copy()
    delta_z_cube.rename('height level widths')

    zdim = z_coord.shape[0]

    for i in range(0, zdim):
        # Distance to lower bound
        if i == zdim - 1:
            # bottom level
            dist_to_lower_bound = 0.5 * (geopot_cube.data[:, i - 1, :, :] -
                                         geopot_cube.data[:, i, :, :])
        else:
            dist_to_lower_bound = 0.5 * (geopot_cube.data[:, i, :, :] -
                                         geopot_cube.data[:, i + 1, :, :])

        if i == 0:
            dist_to_upper_bound = 0.5 * (geopot_cube.data[:, i, :, :] -
                                         geopot_cube.data[:, i + 1, :, :])
        else:
            dist_to_upper_bound = 0.5 * (geopot_cube.data[:, i - 1, :, :] -
                                         geopot_cube.data[:, i, :, :])

        delta_z_cube.data[:,
                          i, :, :] = dist_to_lower_bound + dist_to_upper_bound

    delta_z_cube = delta_z_cube / STANDARD_GRAVITY
    delta_z_cube.units = geopot_cube.units / STANDARD_GRAVITY_UNIT

    return delta_z_cube
