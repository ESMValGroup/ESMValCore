"""Derivation of variable `AIRC_NO_s`."""

"""The variable 'AIRC_NO_s' is an EMAC variable that is used """
"""for monitoring EMAC output. It is here integrated over all """
"""available levels (with the help of the fields 'geopot_ave' """
"""and 'geosp_ave'. """
"""The variable is stored in the EMAC CMIP6 channel 'import_grid'. """
"""AIRC_NO_s: Aircraft NO, summed """


import iris
import iris.analysis
from . import var_name_constraint
import cf_units
from scipy import constants


def derive(cubes):
    no_cube = cubes.extract_strict(var_name_constraint('airc_NO'))
    geopot_cube = cubes.extract_strict(var_name_constraint('geopot_ave'))
    geosp_cube = cubes.extract_strict(var_name_constraint('geosp_ave'))

    z_coord = no_cube.coords(dimensions=1)
    layer_widths = _level_widths(no_cube, geopot_cube, geosp_cube, z_coord, top=100000.)

    no_s_cube = (no_cube * layer_widths)
    no_s_cube = no_s_cube.collapsed(z_coord_name, iris.analysis.SUM)
    no_s_cube.units = (no_cube.units * layer_widths.units)


# Helper functions
def _level_widths(no_cube, geopot_cube, geosp_cube, z_coord, top=100000.):
    """Create a cube with pressure level widths.
    This is done by taking a 2D surface pressure field as lower bound.
    Parameters
    ----------
        no_cube: aircraft NOx emissions (given on model levels)
        geopot_cube: geopotential height (m^2 s^-2)
        geosp_cube: geopotential height at surface (m^2 s-^2)
        z_coord: vertical coordinate (model levels)
        top: top of highest level (m)
    Returns
    -------
    iris.cube.Cube
    `Cube` of same shape as `no_cube` containing pressure level widths (Pa).
    """
    # Constants
    STANDARD_GRAVITY = 9.81
    STANDARD_GRAVITY_UNIT = cf_units.Unit('m s^-2')

    delta_z_cube = no_cube
    delta_z_cube.data = geopot_cube.data / STANDARD_GRAVITY
    delta_z_cube.rename('height level widths')
    delta_z_cube.units = (geopot_cube.units / STANDARD_GRAVITY_UNIT)

    for i in range(0, len(z_coord)):
        # Distance to lower bound
        if i == len(z_coord) - 1:
            # bottom level
            dist_to_lower_bound = geopot_cube.data[:,i,:,:] - geosp_cube.data[:,:,:]
        else:
            dist_to_lower_bound = 0.5 * (geopot_cube.data[:,i,:,:] - geopot_cube.data[:,i+1,:,:])

        if i == 0:
            dist_to_upper_bound = top - geopot_cube.data[:,i,:,:]
        else:
            dist_to_upper_bound = 0.5 * (geopot_cube.data[:,i-1,:,:] - geopot_cube.data[:,i,:,:])

        delta_z_cube.data[:,i,:,:] = dist_to_lower_bound + dist_to_upper_bound

    return delta_z_cube
