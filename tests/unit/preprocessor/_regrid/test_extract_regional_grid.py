"""Unit test for :func:`esmvalcore.preprocessor._regrid`"""

import numpy as np
import pytest

from esmvalcore.preprocessor._regrid import (
    _spec_to_latlonvals,
    _stock_global_cube,
    regrid,
)

SPEC_KEYS = 'xfirst', 'xsize', 'xinc', 'yfirst', 'ysize', 'yinc'
PASSING_SPECS = (dict(zip(SPEC_KEYS, spec)) for spec in (
    (0, 360, 1, -90, 180, 1),
    (0, 360, 10, -90, 180, 10),
    (0, 360, 20, -90, 180, 20),
    (90, 30, 5, 30, 60, 5),
    (125, 35, 10, -45, 65, 5),
    (3, 30, 6, -87, 60, 6),
))

FAILING_SPECS = (dict(zip(SPEC_KEYS, spec)) for spec in (
    (90, 30, 5, -130, 60, 5),
    (90, 30, 5, 30, 360, 5),
    (-90, 30, 5, 30, 60, 5),
    (90, 720, 5, 30, 60, 5),
    (90, 30, 5, 30, -60, 5),
    (90, -30, 5, 30, 60, 5),
    (90, 30, -5, 30, 60, 5),
    (90, 30, 5, 30, 60, -5),
    (90, 0, 5, 30, 60, 5),
    (90, 30, 5, 30, 0, 5),
    (90, 30, 0, 30, 60, 5),
    (90, 30, 5, 30, 60, 0),
))


@pytest.mark.parametrize('spec', PASSING_SPECS)
def test_extract_regional_grid_passing(spec):
    """Test regridding with regional target spec."""
    global_cube = _stock_global_cube('10x10')
    scheme = 'linear'

    result_cube = regrid(global_cube, target_grid=spec, scheme=scheme)

    expected_latvals, expected_lonvals = _spec_to_latlonvals(**spec)

    latvals = result_cube.coord('latitude').points
    lonvals = result_cube.coord('longitude').points

    np.testing.assert_array_equal(latvals, expected_latvals)
    np.testing.assert_array_equal(lonvals, expected_lonvals)


@pytest.mark.parametrize('spec', FAILING_SPECS)
def test_extract_regional_grid_failing(spec):
    """Test failing input for spec."""
    global_cube = _stock_global_cube('10x10')
    scheme = 'linear'

    with pytest.raises(ValueError):
        _ = regrid(global_cube, target_grid=spec, scheme=scheme)
