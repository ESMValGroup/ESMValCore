"""Volume and z coordinate operations on data cubes.

Allows for selecting data subsets using certain volume bounds; selecting
depth or height regions; constructing volumetric averages;
"""
import logging

import dask.array as da
import iris
import numpy as np

from ._shared import get_iris_analysis_operation, operator_accept_weights

logger = logging.getLogger(__name__)


def extract_volume(cube, z_min, z_max):
    """Subset a cube based on a range of values in the z-coordinate.

    Function that subsets a cube on a box (z_min, z_max)
    This function is a restriction of masked_cube_lonlat();
    Note that this requires the requested z-coordinate range to be the
    same sign as the iris cube. ie, if the cube has z-coordinate as
    negative, then z_min and z_max need to be negative numbers.

    Parameters
    ----------
    cube: iris.cube.Cube
        input cube.
    z_min: float
        minimum depth to extract.
    z_max: float
        maximum depth to extract.

    Returns
    -------
    iris.cube.Cube
        z-coord extracted cube.
    """
    if z_min > z_max:
        # minimum is below maximum, so switch them around
        zmax = float(z_min)
        zmin = float(z_max)
    else:
        zmax = float(z_max)
        zmin = float(z_min)

    z_constraint = iris.Constraint(
        coord_values={
            cube.coord(axis='Z'): lambda cell: zmin < cell.point < zmax
        })

    return cube.extract(z_constraint)


def calculate_volume(cube):
    """Calculate volume from a cube.

    This function is used when the volume netcdf fx_variables can't be found.

    Parameters
    ----------
    cube: iris.cube.Cube
        input cube.

    Returns
    -------
    float
        grid volume.
    """
    # ####
    # Load depth field and figure out which dim is which.
    depth = cube.coord(axis='z')
    z_dim = cube.coord_dims(cube.coord(axis='z'))[0]

    # ####
    # Load z direction thickness
    thickness = depth.bounds[..., 1] - depth.bounds[..., 0]

    # ####
    # Calculate grid volume:
    area = da.array(iris.analysis.cartography.area_weights(cube))
    if thickness.ndim == 1 and z_dim == 1:
        grid_volume = area * thickness[None, :, None, None]
    if thickness.ndim == 4 and z_dim == 1:
        grid_volume = area * thickness[:, :]

    return grid_volume


def volume_statistics(cube, operator):
    """Apply a statistical operation over a volume.

    The volume average is weighted according to the cell volume. Cell volume
    is calculated from iris's cartography tool multiplied by the cell
    thickness.

    Parameters
    ----------
    cube: iris.cube.Cube
        Input cube.
    operator: str
        The operation to apply to the cube, options are: 'mean'.

    Returns
    -------
    iris.cube.Cube
        collapsed cube.

    Raises
    ------
    ValueError
        if input cube shape differs from grid volume cube shape.
    """
    # TODO: Test sigma coordinates.
    # TODO: Add other operations.

    if operator != 'mean':
        raise ValueError(f'Volume operator {operator} not recognised.')

    try:
        grid_volume = cube.cell_measure('ocean_volume').core_data()
    except iris.exceptions.CellMeasureNotFoundError:
        logger.debug('Cell measure "ocean_volume" not found in cube. '
                     'Check fx_file availability.')
        logger.debug('Attempting to calculate grid cell volume...')
        grid_volume = calculate_volume(cube)
    else:
        grid_volume = da.broadcast_to(grid_volume, cube.shape)

    if cube.data.shape != grid_volume.shape:
        raise ValueError('Cube shape ({}) doesn`t match grid volume shape '
                         f'({cube.shape, grid_volume.shape})')

    masked_volume = da.ma.masked_where(
        da.ma.getmaskarray(cube.lazy_data()),
        grid_volume)
    result = cube.collapsed(
        [cube.coord(axis='Z'), cube.coord(axis='Y'), cube.coord(axis='X')],
        iris.analysis.MEAN,
        weights=masked_volume)

    return result


def axis_statistics(cube, axis, operator):
    """Perform statistics along a given axis.

    Operates over an axis direction. If weights are required,
    they are computed using the coordinate bounds.

    Arguments
    ---------
    cube: iris.cube.Cube
        Input cube.
    axis: str
        Direction over where to apply the operator. Possible values
        are 'x', 'y', 'z', 't'.
    operator: str
        Statistics to perform. Available operators are:
        'mean', 'median', 'std_dev', 'sum', 'variance',
        'min', 'max', 'rms'.

    Returns
    -------
    iris.cube.Cube
        collapsed cube.
    """
    try:
        coord = cube.coord(axis=axis)
    except iris.exceptions.CoordinateNotFoundError as err:
        raise ValueError(
            'Axis {} not found in cube {}'.format(
                axis,
                cube.summary(shorten=True))) from err
    coord_dims = cube.coord_dims(coord)
    if len(coord_dims) > 1:
        raise NotImplementedError(
            'axis_statistics not implemented for '
            'multidimensional coordinates.')
    operation = get_iris_analysis_operation(operator)
    if operator_accept_weights(operator):
        coord_dim = coord_dims[0]
        expand = list(range(cube.ndim))
        expand.remove(coord_dim)
        bounds = coord.core_bounds()
        weights = np.abs(bounds[..., 1] - bounds[..., 0])
        weights = np.expand_dims(weights, expand)
        weights = da.broadcast_to(weights, cube.shape)
        result = cube.collapsed(coord,
                                operation,
                                weights=weights)
    else:
        result = cube.collapsed(coord, operation)

    return result


def depth_integration(cube):
    """Determine the total sum over the vertical component.

    Requires a 3D cube. The z-coordinate
    integration is calculated by taking the sum in the z direction of the
    cell contents multiplied by the cell thickness.

    Arguments
    ---------
    cube: iris.cube.Cube
        input cube.

    Returns
    -------
    iris.cube.Cube
        collapsed cube.
    """
    result = axis_statistics(cube, axis='z', operator='sum')
    result.rename('Depth_integrated_' + str(cube.name()))
    # result.units = Unit('m') * result.units # This doesn't work:
    # TODO: Change units on cube to reflect 2D concentration (not 3D)
    # Waiting for news from iris community.
    return result


def extract_transect(cube, latitude=None, longitude=None):
    """Extract data along a line of constant latitude or longitude.

    Both arguments, latitude and longitude, are treated identically.
    Either argument can be a single float, or a pair of floats, or can be
    left empty.
    The single float indicates the latitude or longitude along which the
    transect should be extracted.
    A pair of floats indicate the range that the transect should be
    extracted along the secondairy axis.

    For instance `'extract_transect(cube, longitude=-28)'` will produce a
    transect along 28 West.

    Also, `'extract_transect(cube, longitude=-28, latitude=[-50, 50])'` will
    produce a transect along 28 West  between 50 south and 50 North.

    This function is not yet implemented for irregular arrays - instead
    try the extract_trajectory function, but note that it is currently
    very slow. Alternatively, use the regrid preprocessor to regrid along
    a regular grid and then extract the transect.

    Parameters
    ----------
    cube: iris.cube.Cube
        input cube.
    latitude: None, float or [float, float], optional
        transect latiude or range.
    longitude:  None, float or [float, float], optional
        transect longitude or range.

    Returns
    -------
    iris.cube.Cube
        collapsed cube.

    Raises
    ------
    ValueError
        slice extraction not implemented for irregular grids.
    ValueError
        latitude and longitude are both floats or lists; not allowed
        to slice on both axes at the same time.
    """
    # ###
    coord_dim2 = False
    second_coord_range = False
    lats = cube.coord('latitude')
    lons = cube.coord('longitude')

    if lats.ndim == 2:
        raise ValueError(
            'extract_transect: Not implemented for irregular arrays!' +
            '\nTry regridding the data first.')

    if isinstance(latitude, float) and isinstance(longitude, float):
        raise ValueError(
            "extract_transect: Can't slice along lat and lon at the same time")

    if isinstance(latitude, list) and isinstance(longitude, list):
        raise ValueError(
            "extract_transect: Can't reduce lat and lon at the same time")

    for dim_name, dim_cut, coord in zip(['latitude', 'longitude'],
                                        [latitude, longitude], [lats, lons]):
        # ####
        # Look for the first coordinate.
        if isinstance(dim_cut, float):
            coord_index = coord.nearest_neighbour_index(dim_cut)
            coord_dim = cube.coord_dims(dim_name)[0]

        # ####
        # Look for the second coordinate.
        if isinstance(dim_cut, list):
            coord_dim2 = cube.coord_dims(dim_name)[0]
            second_coord_range = [
                coord.nearest_neighbour_index(dim_cut[0]),
                coord.nearest_neighbour_index(dim_cut[1])
            ]
    # ####
    # Extracting the line of constant longitude/latitude
    slices = [slice(None) for i in cube.shape]
    slices[coord_dim] = coord_index

    if second_coord_range:
        slices[coord_dim2] = slice(second_coord_range[0],
                                   second_coord_range[1])
    return cube[tuple(slices)]


def extract_trajectory(cube, latitudes, longitudes, number_points=2):
    """Extract data along a trajectory.

    latitudes and longitudes are the pairs of coordinates for two points.
    number_points is the number of points between the two points.

    This version uses the expensive interpolate method, but it may be
    necceasiry for irregular grids.

    If only two latitude and longitude coordinates are given,
    extract_trajectory will produce a cube will extrapolate along a line
    between those two points, and will add `number_points` points between
    the two corners.

    If more than two points are provided, then
    extract_trajectory will produce a cube which has extrapolated the data
    of the cube to those points, and `number_points` is not needed.

    Parameters
    ----------
    cube: iris.cube.Cube
        input cube.
    latitudes: list
        list of latitude coordinates (floats).
    longitudes: list
        list of longitude coordinates (floats).
    number_points: int
        number of points to extrapolate (optional).

    Returns
    -------
    iris.cube.Cube
        collapsed cube.

    Raises
    ------
    ValueError
        if latitude and longitude have different dimensions.
    """
    from iris.analysis.trajectory import interpolate

    if len(latitudes) != len(longitudes):
        raise ValueError(
            'Longitude & Latitude coordinates have different lengths')

    if len(latitudes) == len(longitudes) == 2:
        minlat, maxlat = np.min(latitudes), np.max(latitudes)
        minlon, maxlon = np.min(longitudes), np.max(longitudes)

        longitudes = np.linspace(minlon, maxlon, num=number_points)
        latitudes = np.linspace(minlat, maxlat, num=number_points)

    points = [('latitude', latitudes), ('longitude', longitudes)]
    interpolated_cube = interpolate(cube, points)  # Very slow!
    return interpolated_cube
