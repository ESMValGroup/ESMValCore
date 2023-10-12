"""Auxiliary derivation functions used for multiple variables."""

import logging
from copy import deepcopy

import iris
import numpy as np
from iris import NameConstraint
from scipy import constants

logger = logging.getLogger(__name__)


def cloud_area_fraction(cubes, tau_constraint, plev_constraint):
    """Calculate cloud area fraction for different parameters."""
    clisccp_cube = cubes.extract_cube(NameConstraint(var_name='clisccp'))
    new_cube = clisccp_cube
    new_cube = new_cube.extract(tau_constraint & plev_constraint)
    coord_names = [
        coord.standard_name for coord in new_cube.coords()
        if len(coord.points) > 1
    ]
    if 'atmosphere_optical_thickness_due_to_cloud' in coord_names:
        new_cube = new_cube.collapsed(
            'atmosphere_optical_thickness_due_to_cloud', iris.analysis.SUM)
    if 'air_pressure' in coord_names:
        new_cube = new_cube.collapsed('air_pressure', iris.analysis.SUM)

    return new_cube


def column_average(cube, hus_cube, zg_cube, ps_cube):
    """Calculate column-averaged mole fractions.

    The calculation follows the method described in the obs4MIPs technical note
    "Merged SCIMACHY/ENVISAT and TANSO-FTS/GOSAT atmospheric column-average
    dry-air mole fraction of CH4 (XCH4)".
    by M. Buchwitz and M. Reuter, available at:
    http://esgf-data1.ceda.ac.uk/thredds/fileServer/esg_obs4mips/
        esacci/ghg/data/obs4mips/crdp_3/CH4/v100/
        TechNote_SCIAGOSAT_L3_CRDP3_001_XCH4_FINAL.pdf

    Parameters
    ----------
    cube : iris.cube.Cube
        Cube containing a mole fraction of a gas (must be 4D with coordinates
        ``time``, ``air_pressure``, ``latitude`` and ``longitude``).
        ``latitude`` and ``air_pressure``).
    hus_cube : iris.cube.Cube
        Cube containing ``specific_humidity`` (must be 4D with coordinates
        ``time``, ``air_pressure``, ``latitude`` and ``longitude``).
        ``latitude`` and ``air_pressure``).
    zg_cube : iris.cube.Cube
        Cube containing ``geopotential_height`` (must be 3D with coordinates
        ``time``, ``latitude`` and ``longitude``).
    ps_cube : iris.cube.Cube
        Cube containing ``surface_air_pressure`` (must be 3D with coordinates
        ``time``, ``latitude`` and ``longitude``).

    Returns
    -------
    iris.cube.Cube
        Column-averaged mole fraction of a gas.

    """
    # Convert units of data
    hus_cube.convert_units('1')
    zg_cube.convert_units('m')
    ps_cube.convert_units('Pa')

    # Level thickness (note: Buchwitz & Reuter use hPa but we use Pa; in fact,
    # this does not matter as units cancel out when calculating column-average
    p_layer_widths = pressure_level_widths(cube, ps_cube, top_limit=0.0)

    # Latitudes (1-dim array)
    lat = cube.coord('latitude').points

    # Gravitational acceleration g_0 on the geoid approximated by the
    # international gravity formula depending only on the latitude
    g_0 = np.array(lat)
    g_0 = 9.780327 * (1.0 + 0.0053024 * (np.sin(lat / 180.0 * np.pi))**2 -
                      0.0000058 * (np.sin(2.0 * lat / 180.0 * np.pi))**2)

    # Approximation of the gravitational acceleration including the
    # free air correction
    # Note: the formula for g given in Buchwitz & Reuter contains a typo and
    # should read: g_0**2 - 2*f*zg (i.e. minus instead of +)
    fair_cor = 3.0825958e-6
    g_4d_array = iris.util.broadcast_to_shape(g_0**2, zg_cube.shape, [2])
    g_4d_array = np.sqrt(g_4d_array.data - 2.0 * fair_cor * zg_cube.data)

    # Number of dry air particles (air molecules excluding water vapor) within
    # each layer
    mw_air = 28.9644e-3
    n_dry = ((hus_cube * -1.0 + 1.0) * constants.value('Avogadro constant') *
             p_layer_widths.data / (mw_air * g_4d_array))

    # Number of gas molecules per layer
    cube.data = cube.core_data() * n_dry.core_data()

    # Column-average
    cube = cube.collapsed('air_pressure', iris.analysis.SUM)
    cube.data = (
        cube.core_data() /
        n_dry.collapsed('air_pressure', iris.analysis.SUM).core_data())
    return cube


def pressure_level_widths(cube, ps_cube, top_limit=0.0):
    """Create a cube with pressure level widths.

    This is done by taking a 2D surface pressure field as lower bound.

    Parameters
    ----------
    cube : iris.cube.Cube
        Cube containing ``air_pressure`` coordinate.
    ps_cube : iris.cube.Cube
        Cube containing ``surface_air_pressure``.
    top_limit : float
        Pressure in Pa.

    Returns
    -------
    iris.cube.Cube
        Cube of same shape as ``cube`` containing pressure level widths.

    """
    pressure_array = _create_pressure_array(cube, ps_cube, top_limit)

    data = _get_pressure_level_widths(pressure_array)
    p_level_widths_cube = cube.copy(data=data)
    p_level_widths_cube.rename('pressure level widths')
    p_level_widths_cube.units = ps_cube.units

    return p_level_widths_cube


def _create_pressure_array(cube, ps_cube, top_limit):
    """Create an array filled with the ``air_pressure`` coord values.

    The array is created from ``cube`` with the same dimensions as ``cube``.
    This array is then sandwiched with a 2D array containing the surface
    pressure and a 2D array containing the top pressure limit.

    """
    # Create 4D array filled with pressure level values
    p_levels = cube.coord('air_pressure').points.astype(np.float32)
    p_4d_array = iris.util.broadcast_to_shape(p_levels, cube.shape, [1])

    # Create 4d array filled with surface pressure values
    shape = cube.shape
    ps_4d_array = iris.util.broadcast_to_shape(ps_cube.data, shape, [0, 2, 3])

    # Set pressure levels below the surface pressure to NaN
    pressure_4d = np.where((ps_4d_array - p_4d_array) < 0, np.NaN, p_4d_array)

    # Make top_limit last pressure level
    top_limit_array = np.full(ps_cube.shape, top_limit, dtype=np.float32)
    data = top_limit_array[:, np.newaxis, :, :]
    pressure_4d = np.concatenate((pressure_4d, data), axis=1)

    # Make surface pressure the first pressure level
    data = ps_cube.data[:, np.newaxis, :, :]
    pressure_4d = np.concatenate((data, pressure_4d), axis=1)

    return pressure_4d


def _get_pressure_level_widths(array, air_pressure_axis=1):
    """Compute pressure level widths.

    For a 1D array with pressure level columns, return a 1D array with
    pressure level widths.

    """
    array = np.copy(array)
    if np.any(np.diff(array, axis=air_pressure_axis) > 0.0):
        raise ValueError("Pressure level value increased with height")

    # Calculate centers
    indices = [slice(None)] * array.ndim
    array_shifted = np.roll(array, -1, axis=air_pressure_axis)
    index_0 = deepcopy(indices)
    index_0[air_pressure_axis] = 0
    array_shifted[tuple(index_0)] = array[tuple(index_0)]
    index_neg1 = deepcopy(indices)
    index_neg1[air_pressure_axis] = -1
    index_neg2 = deepcopy(indices)
    index_neg2[air_pressure_axis] = -2
    array[tuple(index_neg2)] = array[tuple(index_neg1)]
    array_centers = (array + array_shifted) / 2.0
    index_range_neg1 = deepcopy(indices)
    index_range_neg1[air_pressure_axis] = slice(None, -1)
    array_centers = array_centers[tuple(index_range_neg1)]

    # Remove NaNs (replace them with surface pressure)
    dim_map = np.arange(array_centers.ndim)
    dim_map = np.delete(dim_map, air_pressure_axis)
    array_centers_surface = iris.util.broadcast_to_shape(
        array_centers[tuple(index_0)], array_centers.shape, dim_map)
    array_centers = np.where(np.isnan(array_centers), array_centers_surface,
                             array_centers)

    # Calculate level widths
    p_level_widths = -np.diff(array_centers, axis=air_pressure_axis)
    return p_level_widths
