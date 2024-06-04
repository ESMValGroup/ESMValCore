"""Volume and z coordinate operations on data cubes.

Allows for selecting data subsets using certain volume bounds; selecting
depth or height regions; constructing volumetric averages;
"""
from __future__ import annotations

import logging
import warnings
from typing import Iterable, Literal, Optional, Sequence

import dask.array as da
import iris
import numpy as np
from iris.coords import AuxCoord, CellMeasure
from iris.cube import Cube

from ._shared import (
    get_iris_aggregator,
    get_normalized_cube,
    preserve_float_dtype,
    try_adding_calculated_cell_area,
    update_weights_kwargs,
)
from ._supplementary_vars import register_supplementaries

logger = logging.getLogger(__name__)


def extract_volume(
    cube: Cube,
    z_min: float,
    z_max: float,
    interval_bounds: str = 'open',
    nearest_value: bool = False,
) -> Cube:
    """Subset a cube based on a range of values in the z-coordinate.

    Function that subsets a cube on a box of (z_min, z_max),
    (z_min, z_max], [z_min, z_max) or [z_min, z_max]
    Note that this requires the requested z-coordinate range to be the
    same sign as the iris cube. ie, if the cube has z-coordinate as
    negative, then z_min and z_max need to be negative numbers.
    If nearest_value is set to `False`, the extraction will be
    performed with the given z_min and z_max values.
    If nearest_value is set to `True`, the cube extraction will be
    performed taking into account the z_coord values that are closest
    to the z_min and z_max values.

    Parameters
    ----------
    cube:
        Input cube.
    z_min:
        Minimum depth to extract.
    z_max:
        Maximum depth to extract.
    interval_bounds:
        Sets left bound of the interval to either 'open', 'closed',
        'left_closed' or 'right_closed'.
    nearest_value:
        Extracts considering the nearest value of z-coord to z_min and z_max.

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

    z_coord = cube.coord(axis='Z')

    if nearest_value:
        min_index = np.argmin(np.abs(z_coord.core_points() - zmin))
        max_index = np.argmin(np.abs(z_coord.core_points() - zmax))
        zmin = z_coord.core_points()[min_index]
        zmax = z_coord.core_points()[max_index]

    if interval_bounds == 'open':
        coord_values = {z_coord: lambda cell: zmin < cell.point < zmax}
    elif interval_bounds == 'closed':
        coord_values = {z_coord: lambda cell: zmin <= cell.point <= zmax}
    elif interval_bounds == 'left_closed':
        coord_values = {z_coord: lambda cell: zmin <= cell.point < zmax}
    elif interval_bounds == 'right_closed':
        coord_values = {z_coord: lambda cell: zmin < cell.point <= zmax}
    else:
        raise ValueError(
            'Depth extraction bounds can be set to "open", "closed", '
            f'"left_closed", or "right_closed". Got "{interval_bounds}".')

    z_constraint = iris.Constraint(coord_values=coord_values)

    return cube.extract(z_constraint)


def calculate_volume(cube: Cube) -> da.core.Array:
    """Calculate volume from a cube.

    This function is used when the 'ocean_volume' cell measure can't be found.
    The output data will be given in cubic meters (m3).

    Note
    ----
    It gets the cell_area from the cube if it is available. If not, it
    calculates it from the grid. This only works if the grid cell areas can
    be calculated (i.e., latitude and longitude are 1D). The depth coordinate
    units should be convertible to meters.

    Parameters
    ----------
    cube: iris.cube.Cube
        input cube.

    Returns
    -------
    dask.array.core.Array
        Grid volumes.

    """
    # Load depth field and figure out which dim is which
    depth = cube.coord(axis='z')
    z_dim = cube.coord_dims(depth)
    depth = depth.copy()

    # Assert z has length > 0
    if not z_dim:
        raise ValueError("Cannot compute volume with scalar Z-axis")

    # Guess bounds if missing
    if not depth.has_bounds():
        depth.guess_bounds()
    if depth.core_bounds().shape[-1] != 2:
        raise ValueError(
            f"Z axis bounds shape found {depth.core_bounds().shape}. "
            "Bounds should be 2 in the last dimension to compute the "
            "thickness.")

    # Convert units to get the thickness in meters
    try:
        depth.convert_units('m')
    except ValueError as err:
        raise ValueError(
            f'Cannot compute volume using the Z-axis. {err}') from err

    # Calculate Z-direction thickness
    thickness = depth.core_bounds()[..., 1] - depth.core_bounds()[..., 0]

    # Get or calculate the horizontal areas of the cube
    has_cell_measure = bool(cube.cell_measures('cell_area'))
    try_adding_calculated_cell_area(cube)
    area = cube.cell_measure('cell_area').copy()
    area_dim = cube.cell_measure_dims(area)

    # Ensure cell area is in square meters as the units
    area.convert_units('m2')

    # Make sure input cube has not been modified
    if not has_cell_measure:
        cube.remove_cell_measure('cell_area')

    area_arr = iris.util.broadcast_to_shape(
        area.core_data(), cube.shape, area_dim)
    thickness_arr = iris.util.broadcast_to_shape(
        thickness, cube.shape, z_dim)
    grid_volume = area_arr * thickness_arr

    return grid_volume


def _try_adding_calculated_ocean_volume(cube: Cube) -> None:
    """Try to add calculated cell measure 'ocean_volume' to cube (in-place)."""
    if cube.cell_measures('ocean_volume'):
        return

    logger.debug(
        "Found no cell measure 'ocean_volume' in cube %s. Check availability "
        "of supplementary variables",
        cube.summary(shorten=True),
    )
    logger.debug("Attempting to calculate grid cell volume")

    grid_volume = calculate_volume(cube)

    cell_measure = CellMeasure(
        grid_volume,
        standard_name='ocean_volume',
        units='m3',
        measure='volume',
    )
    cube.add_cell_measure(cell_measure, np.arange(cube.ndim))


@register_supplementaries(
    variables=['volcello', 'areacello'],
    required='prefer_at_least_one',
)
@preserve_float_dtype
def volume_statistics(
    cube: Cube,
    operator: str,
    normalize: Optional[Literal['subtract', 'divide']] = None,
    **operator_kwargs,
) -> Cube:
    """Apply a statistical operation over a volume.

    The volume average is weighted according to the cell volume.

    Parameters
    ----------
    cube:
        Input cube. The input cube should have a
        :class:`iris.coords.CellMeasure` named ``'ocean_volume'``, unless it
        has a :class:`iris.coords.CellMeasure` named ``'cell_area'`` or
        regular 1D latitude and longitude coordinates so the cell areas
        can be computed using :func:`iris.analysis.cartography.area_weights`.
        The volume will be computed from the area multiplied by the
        thickness, computed from the bounds of the vertical coordinate.
        In that case, vertical coordinate units should be convertible to
        meters.
    operator:
        The operation. Used to determine the :class:`iris.analysis.Aggregator`
        object used to calculate the statistics. Currently, only `mean` is
        allowed.
    normalize:
        If given, do not return the statistics cube itself, but rather, the
        input cube, normalized with the statistics cube. Can either be
        `subtract` (statistics cube is subtracted from the input cube) or
        `divide` (input cube is divided by the statistics cube).
    **operator_kwargs:
        Optional keyword arguments for the :class:`iris.analysis.Aggregator`
        object defined by `operator`.

    Note
    ----
    This preprocessor has been designed for oceanic variables, but it might
    be applicable to atmospheric data as well.

    Returns
    -------
    iris.cube.Cube
        Collapsed cube.

    """
    has_cell_measure = bool(cube.cell_measures('ocean_volume'))

    # TODO: Test sigma coordinates.
    # TODO: Add other operations.
    if operator != 'mean':
        raise ValueError(f"Volume operator {operator} not recognised.")
    # get z, y, x coords
    z_axis = cube.coord(axis='Z')
    y_axis = cube.coord(axis='Y')
    x_axis = cube.coord(axis='X')

    # assert z axis only uses 1 dimension more than x, y axis
    xy_dims = tuple({*cube.coord_dims(y_axis), *cube.coord_dims(x_axis)})
    xyz_dims = tuple({*cube.coord_dims(z_axis), *xy_dims})
    if len(xyz_dims) > len(xy_dims) + 1:
        raise ValueError(
            f"X and Y axis coordinates depend on {xy_dims} dimensions, "
            f"while X, Y, and Z axis depends on {xyz_dims} dimensions. "
            "This may indicate Z axis depending on other dimension than "
            "space that could provoke invalid aggregation...")

    (agg, agg_kwargs) = get_iris_aggregator(operator, **operator_kwargs)
    agg_kwargs = update_weights_kwargs(
        agg,
        agg_kwargs,
        'ocean_volume',
        cube,
        _try_adding_calculated_ocean_volume,
    )

    result = cube.collapsed([z_axis, y_axis, x_axis], agg, **agg_kwargs)
    if normalize is not None:
        result = get_normalized_cube(cube, result, normalize)

    # Make sure input cube has not been modified
    if not has_cell_measure and cube.cell_measures('ocean_volume'):
        cube.remove_cell_measure('ocean_volume')

    return result


@preserve_float_dtype
def axis_statistics(
    cube: Cube,
    axis: str,
    operator: str,
    normalize: Optional[Literal['subtract', 'divide']] = None,
    **operator_kwargs,
) -> Cube:
    """Perform statistics along a given axis.

    Operates over an axis direction.

    Note
    ----
    The `mean`, `sum` and `rms` operations are weighted by the corresponding
    coordinate bounds by default. For `sum`, the units of the resulting cube
    will be multiplied by corresponding coordinate units.

    Arguments
    ---------
    cube:
        Input cube.
    axis:
        Direction over where to apply the operator. Possible values are `x`,
        `y`, `z`, `t`.
    operator:
        The operation. Used to determine the :class:`iris.analysis.Aggregator`
        object used to calculate the statistics. Allowed options are given in
        :ref:`this table <supported_stat_operator>`.
    normalize:
        If given, do not return the statistics cube itself, but rather, the
        input cube, normalized with the statistics cube. Can either be
        `subtract` (statistics cube is subtracted from the input cube) or
        `divide` (input cube is divided by the statistics cube).
    **operator_kwargs:
        Optional keyword arguments for the :class:`iris.analysis.Aggregator`
        object defined by `operator`.

    Returns
    -------
    iris.cube.Cube
        Collapsed cube.

    """
    (agg, agg_kwargs) = get_iris_aggregator(operator, **operator_kwargs)

    # Check if a coordinate for the desired axis exists
    try:
        coord = cube.coord(axis=axis)
    except iris.exceptions.CoordinateNotFoundError as err:
        raise ValueError(
            f"Axis {axis} not found in cube {cube.summary(shorten=True)}"
        ) from err

    # Multidimensional coordinates are currently not supported
    coord_dims = cube.coord_dims(coord)
    if len(coord_dims) > 1:
        raise NotImplementedError(
            "axis_statistics not implemented for multidimensional "
            "coordinates."
        )

    # For weighted operations, create a dummy weights coordinate using the
    # bounds of the original coordinate (this handles units properly, e.g., for
    # sums)
    agg_kwargs = update_weights_kwargs(
        agg,
        agg_kwargs,
        '_axis_statistics_weights_',
        cube,
        _add_axis_stats_weights_coord,
        coord=coord,
        coord_dims=coord_dims,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message=(
                "Cannot check if coordinate is contiguous: Invalid "
                "operation for '_axis_statistics_weights_'"
            ),
            category=UserWarning,
            module='iris',
        )
        result = cube.collapsed(coord, agg, **agg_kwargs)

    if normalize is not None:
        result = get_normalized_cube(cube, result, normalize)

    # Make sure input and output cubes do not have auxiliary coordinate
    if cube.coords('_axis_statistics_weights_'):
        cube.remove_coord('_axis_statistics_weights_')
    if result.coords('_axis_statistics_weights_'):
        result.remove_coord('_axis_statistics_weights_')

    return result


def _add_axis_stats_weights_coord(cube, coord, coord_dims):
    """Add weights for axis_statistics to cube (in-place)."""
    weights = np.abs(coord.lazy_bounds()[:, 1] - coord.lazy_bounds()[:, 0])
    if not cube.has_lazy_data():
        weights = weights.compute()
    weights_coord = AuxCoord(
        weights,
        long_name='_axis_statistics_weights_',
        units=coord.units,
    )
    cube.add_aux_coord(weights_coord, coord_dims)


def depth_integration(cube: Cube) -> Cube:
    """Determine the total sum over the vertical component.

    Requires a 3D cube. The z-coordinate integration is calculated by taking
    the sum in the z direction of the cell contents multiplied by the cell
    thickness. The units of the resulting cube are multiplied by the
    z-coordinate units.

    Arguments
    ---------
    cube:
        Input cube.

    Returns
    -------
    iris.cube.Cube
        Collapsed cube.

    """
    result = axis_statistics(cube, axis='z', operator='sum')
    result.rename('Depth_integrated_' + str(cube.name()))
    return result


def extract_transect(
    cube: Cube,
    latitude: None | float | Iterable[float] = None,
    longitude: None | float | Iterable[float] = None,
) -> Cube:
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
    cube:
        Input cube.
    latitude: optional
        Transect latitude or range.
    longitude:  optional
        Transect longitude or range.

    Returns
    -------
    iris.cube.Cube
        Collapsed cube.

    Raises
    ------
    ValueError
        Slice extraction not implemented for irregular grids.
    ValueError
        Latitude and longitude are both floats or lists; not allowed to slice
        on both axes at the same time.

    """
    # ###
    coord_dim2 = False
    second_coord_range: None | list = None
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

    if second_coord_range is not None:
        slices[coord_dim2] = slice(second_coord_range[0],
                                   second_coord_range[1])
    return cube[tuple(slices)]


def extract_trajectory(
    cube: Cube,
    latitudes: Sequence[float],
    longitudes: Sequence[float],
    number_points: int = 2,
) -> Cube:
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
    cube:
        Input cube.
    latitudes:
        Latitude coordinates.
    longitudes:
        Longitude coordinates.
    number_points: optional
        Number of points to extrapolate.

    Returns
    -------
    iris.cube.Cube
        Collapsed cube.

    Raises
    ------
    ValueError
        Latitude and longitude have different dimensions.

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
