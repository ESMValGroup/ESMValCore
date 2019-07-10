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
import numba
import numpy as np
from scipy import constants

# Constants
STANDARD_GRAVITY = 9.81
STANDARD_GRAVITY_UNIT = cf_units.Unit('m s^-2')
MW_AIR = 29
MW_AIR_UNIT = cf_units.Unit('g mol^-1')
MW_NO = 46  #units given in the EMAC file as "kg(NO2) m-3 s-1"
MW_NO_UNIT = cf_units.Unit('g mol^-1')


def derive(cubes):
	no_cube = cubes.extract_strict(var_name_constraint('airc_NO'))
	#geopot_cube = cubes.extract_strict(var_name_constraint('geopot_ave'))
	#geosp_cube = cubes.extract_strict(var_name_constraint('geosp_ave'))
	aps_cube = cubes.extract_strict(var_name_constraint('aps_ave'))
	hyai_cube = cubes.extract_strict(var_name_constraint('hyai'))
	hybi_cube = cubes.extract_strict(var_name_constraint('hybi'))
	
    p_layer_widths = _pressure_level_widths(
		no_cube, aps_cube, hyai_cube, hybi_cube, top_limit=0.0)	
	z_coord = no_cube.coords(dimensions=1)
	z_coord_name= z_coord[0].name()
	
    no_s_cube = (
        no_cube * p_layer_widths / STANDARD_GRAVITY * MW_NO / MW_AIR)
    no_s_cube = no_s_cube.collapsed(z_coord_name, iris.analysis.SUM)
    no_s_cube.units = (no_cube.units * p_layer_widths.units /
                          STANDARD_GRAVITY_UNIT * MW_NO_UNIT / MW_AIR_UNIT)

    return no_s_cube	
	

# Helper functions
def _pressure_level_widths(
	no_cube, aps_cube, hyai_cube, hybi_cube, z_coord_name, top_limit=0.0):
    """Create a cube with pressure level widths.
    This is done by taking a 2D surface pressure field as lower bound.
    Parameters
    ----------
        no_cube : iris.cube.Cube
            `Cube` containing `Sum of NOx Aircraft Anthropogenic Emissions`.
        aps_cube : iris.cube.Cube
            `Cube` containing `surface pressure`.
        top_limit : float
            Pressure in Pa.
    Returns
    -------
    iris.cube.Cube
    `Cube` of same shape as `no_cube` containing pressure level widths.
    """
    #pressure_array = _create_pressure_array(
		#no_cube, aps_cube, hyai_cube, hybi_cube, z_coord_name, top_limit)

    #data = _apply_pressure_level_widths(pressure_array)
	data = _get_pressure_level_width(
		no_cube, aps_cube, hyai_cube, hybi_cube, z_coord_name, top_limit)
    p_level_widths_cube = no_cube.copy(data=data)
    p_level_widths_cube.rename('pressure level widths')
    p_level_widths_cube.units = aps_cube.units

    return p_level_widths_cube	


def _get_pressure_level_width(
	no_cube, aps_cube, hyai_cube, hybi_cube, z_coord_name, top_limit):
	# Create 4D array with the same size as "no_cube"
	shape = no_cube.shape
	


def _create_pressure_array(
	no_cube, aps_cube, hyai_cube, hybi_cube, z_coord_name, top_limit):
    """Create an array filled with the `air_pressure` coord values.
    The array is created from the `no_cube` with the same dimensions
    as `no_cube`. This array is then sandwiched with a 2D array
    containing the surface pressure and a 2D array containing the top
    pressure limit.
    """
    # Create 4D array filled with pressure level values
    p_levels = no_cube.coord(z_coord_name).points.astype(np.float32)
    p_4d_array = iris.util.broadcast_to_shape(p_levels, no_cube.shape, [1])

    # Create 4d array filled with surface pressure values
    shape = no_cube.shape
    ps_4d_array = iris.util.broadcast_to_shape(aps_cube.data, shape, [0, 2, 3])

    # Set pressure levels below the surface pressure to NaN
    pressure_4d = np.where((ps_4d_array - p_4d_array) < 0, np.NaN, p_4d_array)

    # Make top_limit last pressure level
    top_limit_array = np.full(aps_cube.shape, top_limit, dtype=np.float32)
    data = top_limit_array[:, np.newaxis, :, :]
    pressure_4d = np.concatenate((pressure_4d, data), axis=1)

    # Make surface pressure the first pressure level
    data = aps_cube.data[:, np.newaxis, :, :]
    pressure_4d = np.concatenate((data, pressure_4d), axis=1)

    return pressure_4d


def _apply_pressure_level_widths(array, level_axis=1):
    """Compute pressure level widths.
    For a 1D array with pressure level columns, return a 1D array with
    pressure level widths.
    """
    return np.apply_along_axis(_p_level_widths, level_axis, array)


@numba.jit()  # ~10x faster
def _p_level_widths(array):
    """Create pressure level widths.
    The array with pressure levels is assumed to be monotonic and the
    values are decreasing.
    The first element is the lower boundary (surface pressure), the last
    value is the upper boundary. Thicknesses are only calculated for the
    values between these boundaries, the returned array, therefore,
    contains two elements less.
    >>> _p_level_widths(np.array([1020, 1000, 700, 500, 5]))
    array([170., 250., 595.], dtype=float32)
    >>> _p_level_widths(np.array([990, np.NaN, 700, 500, 5]))
    array([  0., 390., 595.], dtype=float32)
    """
    surface_pressure = array[0]
    top_limit = array[-1]
    array = array[1:-1]

    p_level_widths = np.full(array.shape, np.NAN, dtype=np.float32)

    last_pressure_level = len(array) - 1
    for i, val in enumerate(array):
        # numba would otherwise initialize it to 0 and
        # hide bugs that would occur in raw Python
        bounds_width = np.NAN
        if np.isnan(val):
            bounds_width = 0
        else:
            # Distance to lower bound
            if i == 0 or np.isnan(array[i - 1]):
                # First pressure level with value
                dist_to_lower_bound = surface_pressure - val
            else:
                dist_to_lower_bound = 0.5 * (array[i - 1] - val)

            # Distance to upper bound
            if i == last_pressure_level:  # last pressure level
                dist_to_upper_bound = val - top_limit
            else:
                dist_to_upper_bound = 0.5 * (val - array[i + 1])

            # Check monotonicity - all distances must be >= 0
            if dist_to_lower_bound < 0.0 or dist_to_upper_bound < 0.0:
                raise ValueError("Pressure level value increased with "
                                 "height.")

            bounds_width = dist_to_lower_bound + dist_to_upper_bound

        p_level_widths[i] = bounds_width
		
    return p_level_widths
