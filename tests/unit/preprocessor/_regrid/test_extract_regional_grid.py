"""Unit test for :func:`esmvalcore.preprocessor._regrid`"""

import numpy as np

from esmvalcore.preprocessor._regrid import (
    _spec_to_latlonvals,
    _stock_global_cube,
    regrid,
)


def test_extract_regional_grid():
    global_cube = _stock_global_cube('10x10')

    scheme = 'linear'

    spec = {
        'xsize': 60,
        'ysize': 30,
        'xfirst': -177,
        'xinc': 6,
        'yfirst': -87,
        'yinc': 6,
    }

    result_cube = regrid(global_cube, target_grid=spec, scheme=scheme)

    expected_latvals, expected_lonvals = _spec_to_latlonvals(**spec)

    latvals = result_cube.coord('latitude').points
    lonvals = result_cube.coord('longitude').points

    np.testing.assert_array_equal(latvals, expected_latvals)
    np.testing.assert_array_equal(lonvals, expected_lonvals)
