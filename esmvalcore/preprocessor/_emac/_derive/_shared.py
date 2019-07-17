"""Shared functions for EMAC variable derivation."""
import iris
import iris.coord_categorisation
import numpy as np
from cf_units import Unit

from . import var_name_constraint

STANDARD_GRAVITY = 9.81
STANDARD_GRAVITY_UNITS = Unit('m s-2')


def sum_over_level(cubes, var_names, scale_factor=1.0, level_idx=1):
    """Perform sum over level coordinate."""
    cube = None
    for var_name in var_names:
        if cube is None:
            cube = cubes.extract_strict(var_name_constraint(var_name))
        else:
            cube += cubes.extract_strict(var_name_constraint(var_name))

    # Scale cube
    cube *= scale_factor

    # Get correct coordinate
    lev_coords = cube.coords(dimensions=level_idx, dim_coords=True)
    if lev_coords:
        cube.remove_coord(lev_coords[0])
    lev_coord = iris.coords.DimCoord(np.arange(cube.shape[level_idx]),
                                     var_name='level',
                                     long_name='level')
    cube.add_dim_coord(lev_coord, level_idx)

    # Sum over coordinate
    cube = cube.collapsed(lev_coord, iris.analysis.SUM)
    return cube


def integrate_vertically(cubes,
                         var_name,
                         scale_factor=1.0,
                         height_var='geopot_ave',
                         height_idx=1):
    """Vertical integration.

    Calculate vertical integral over desired variable (on model levels) by
    multiplying each vertical value with layer thickness (m) and summing up all
    vertical levels. If necessary, the input cube is scaled (e.g. to fix wrong
    units).

    Note
    ----
    The values for `var_name` and `height_var` may contain a different number
    of time steps (monthly aggregation is performed).

    """
    cube = cubes.extract_strict(var_name_constraint(var_name))
    height_cube = cubes.extract_strict(var_name_constraint(height_var))

    # Scale cube
    cube *= scale_factor

    # Calculate monthly means for both cubes
    iris.coord_categorisation.add_categorised_coord(cube, 'year_month', 'time',
                                                    _year_month_category)
    iris.coord_categorisation.add_categorised_coord(height_cube, 'year_month',
                                                    'time',
                                                    _year_month_category)
    cube = cube.aggregated_by('year_month', iris.analysis.MEAN)
    height_cube = height_cube.aggregated_by('year_month', iris.analysis.MEAN)
    cube.remove_coord('year_month')
    height_cube.remove_coord('year_month')

    # Calculate layer thickness (m) and synchronize time coordinate
    layer_widths_cube = _level_widths(height_cube, height_idx=height_idx)
    time_idx = layer_widths_cube.coord_dims('time')
    layer_widths_cube.remove_coord('time')
    layer_widths_cube.add_dim_coord(cube.coord('time'), time_idx)

    # Multiply by layer thickness and sum over all levels
    cube *= layer_widths_cube
    height_coord = cube.coord(dimensions=height_idx)
    cube = cube.collapsed(height_coord, iris.analysis.SUM)

    return cube


def _level_widths(cube, height_idx=1):
    """Create a cube with height level widths.

    Parameters
    ----------
    cube : iris.cube.Cube
        `iris.cube.Cube` containing geopotential height (m2 s-2).
    height_idx : int, optional (default: 1)
        Index of the height coordinate of the cube.

    Returns
    -------
    iris.cube.Cube
        `iris.cube.Cube` of same shape as input `cube` containing height level
        widths (m).

    """
    # Get neighboring height levels
    levels_shifted_right = np.roll(cube.data, 1, axis=height_idx)
    levels_shifted_left = np.roll(cube.data, -1, axis=height_idx)

    # Get distances to lower and upper cell boundaries
    dist_to_lower_bounds = 0.5 * (cube.data - levels_shifted_left)
    dist_to_upper_bounds = 0.5 * (levels_shifted_right - cube.data)

    # Fix values at boundary
    lower_width = np.take(dist_to_lower_bounds, -2, axis=height_idx)
    upper_width = np.take(dist_to_upper_bounds, 1, axis=height_idx)
    lower_idx = [slice(None)] * dist_to_lower_bounds.ndim
    upper_idx = [slice(None)] * dist_to_upper_bounds.ndim
    lower_idx[height_idx] = -1
    upper_idx[height_idx] = 0
    dist_to_lower_bounds[lower_idx] = lower_width
    dist_to_upper_bounds[upper_idx] = upper_width

    # Create cube with new data
    level_widths_cube = cube.copy(dist_to_lower_bounds + dist_to_upper_bounds)

    # Multiply by constant
    level_widths_cube /= STANDARD_GRAVITY
    level_widths_cube.units /= STANDARD_GRAVITY_UNITS
    level_widths_cube.var_name = 'height_level_widths'

    return level_widths_cube


def _year_month_category(coord, value):
    """Assign month and year (YYYY-MM) to coordinate value."""
    date = coord.units.num2date(value)
    return f'{date.year}-{date.month}'
