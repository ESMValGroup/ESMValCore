"""Preprocessor functions that remove trends from the data."""
import logging

import dask.array as da
import scipy.signal

logger = logging.getLogger(__name__)


def detrend(cube, dimension='time', method='linear'):
    """
    Detrend data along a given dimension.

    Parameters
    ----------
    cube: iris.cube.Cube
        input cube.
    dimension: str
        Dimension to detrend
    method: str
        Method to detrend. Available: linear, constant. See documentation of
        'scipy.signal.detrend' for details

    Returns
    -------
    iris.cube.Cube
        Detrended cube
    """
    coord = cube.coord(dimension)
    axis = cube.coord_dims(coord)[0]
    detrended = da.apply_along_axis(
        scipy.signal.detrend,
        axis=axis,
        arr=cube.lazy_data(),
        type=method,
        shape=(cube.shape[axis],)
    )
    return cube.copy(detrended)


def linear_trend(cube, dimension='time'):
    """
    Give the trend along a given dimension by detrending first and returning
    the difference

    Parameters
    ----------
    cube: iris.cube.Cube
        input cube.
    dimension: str
        Dimension to detrend
    method: str
        Method to detrend. Available: linear, constant. See documentation of
        'scipy.signal.detrend' for details

    Returns
    -------
    iris.cube.Cube
        Detrended cube
    """
    cube_detrend = detrend(cube, dimension=dimension, method='linear')
    cube.data -= cube_detrend.data

    # # if frequency not in ['yr', 'mon', 'day']
    # if frequency == 'mon':
    #     frequency = 'month'
    # cube.units = Unit(str(cube.units) + f' {frequency}-1')

    cube.long_name = cube.long_name + " trend"

    return cube


#     import scipy.stats

#     if np.ma.isMaskedArray(cube.data):

#     x_arr = x_arr[~y_arr.mask]
#     y_arr = y_arr[~y_arr.mask]

#     x = cube.data
#     y = cube.coord('time').points

#     m, b, r_val, p_val, std_err = scipy.stats.linregress(x[~x.mask],y[~x.mask])








#         import numpy as np

#     import IPython
#     from traitlets.config import get_config
#     c = get_config()
#     c.InteractiveShellEmbed.colors = "Linux"
#     # from cf_units import Unit

#     # masks should be nans
#     IPython.embed(config=c)

#     # check if the cube is masked and replace by nans
#     if np.ma.isMaskedArray(cube.data):

# find ma mal daten wo das so ist
# la=cube.data.mask.sum(axis=0)
# not(np.any(~((la == 0) | (la == cube.shape[0]))))

# cube.data[:,135:140,200:210]

# import copy
# save = copy.deepcopy(cube)
# cube = copy.deepcopy(save)


# cube = copy.deepcopy(save)
# cube = cube[:,135:137,200:202]
# cube.data.mask[30:35, 0, 0] = True
# cube_detrend = detrend(cube, dimension=dimension, method='linear')
# cube.data -= cube_detrend.data
# first=copy.deepcopy(cube)

# cube = copy.deepcopy(save)
# cube = cube[:,135:137,200:202]
# cube.data.mask[30:35, 0, 0] = True
# cube.data.data[30:35, 0, 0] = cube.data.fill_value
# cube_detrend = detrend(cube, dimension=dimension, method='linear')
# cube.data -= cube_detrend.data
# second=copy.deepcopy(cube)

# cube = copy.deepcopy(save)
# cube = cube[:,135:137,200:202]
# cube.data.mask[30:35, 0, 0] = True
# # cube.data = cube.data.filled(np.nan)
# cube = linear_trend(cube)
# third=copy.deepcopy(cube)



# aggregator = iris.analysis.Aggregator('trend', call_func,
#                                         lazy_func=lazy_func,
#                                         x_data=coord.points)
# cube = cube.collapsed(coord, aggregator)


# trend_arr = da.apply_along_axis(
#     _get_slope, axis, data, x_data, dtype=data.dtype, shape=())
# trend_arr = da.ma.masked_invalid(trend_arr)
