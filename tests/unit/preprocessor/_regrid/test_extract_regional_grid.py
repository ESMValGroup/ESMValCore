"""Unit test for :func:`esmvalcore.preprocessor._regrid`"""

from decimal import Decimal

import numpy as np
import pytest

from esmvalcore.preprocessor._regrid import (
    _global_stock_cube,
    _spec_to_latlonvals,
    regrid,
)

SPEC_KEYS = ('start_longitude', 'end_longitude', 'step_longitude',
             'start_latitude', 'end_latitude', 'step_latitude')
PASSING_SPECS = tuple(
    dict(zip(SPEC_KEYS, spec)) for spec in (
        (0, 360, 5, -90, 90, 5),
        (0, 360, 20, -90, 90, 20),
        (0, 21, 5, -90, 90, 1),
        (0, 360, 5, -90, -70, 5),
        (0, 360, 5, -90, -69, 5),
        (350, 370, 1, -90, 90, 1),
        (-20, -5, 1, -90, 90, 1),
        (100, 50, -5, -90, 90, 5),
        (0, 360, 5, 40, 20, -5),
        (0, 359, 10, -90, 89, 10),
        (0, 0, 5, -90, 90, 5),
        (0, 360, 5, 0, 0, 5),
        (0, 9, 0.1, 45, 54, 0.1),
        (3.75, 11.75, 0.5, 46.25, 52.25, 0.5),
    ))

FAILING_SPECS = tuple(
    dict(zip(SPEC_KEYS, spec)) for spec in (
        # (0, 360, 5, -90, 90, 5),
        (0, 360, 5, -90, 180, 5),
        (0, 360, 5, -180, 90, 5),
        (0, 360, 5, -90, 90, -5),
        (0, 360, -5, -90, 90, 5),
        (0, -360, 5, -90, 90, 5),
        (0, 360, 0, -90, 90, 5),
        (0, 360, 5, -90, 90, 0),
    ))


@pytest.mark.parametrize('spec', PASSING_SPECS)
def test_extract_regional_grid_passing(spec):
    """Test regridding with regional target spec."""
    global_cube = _global_stock_cube('10x10')
    scheme = 'linear'

    result_cube = regrid(global_cube, target_grid=spec, scheme=scheme)

    expected_latvals, expected_lonvals = _spec_to_latlonvals(**spec)

    lat_coord = result_cube.coord('latitude')
    lon_coord = result_cube.coord('longitude')

    np.testing.assert_array_equal(lat_coord.points, expected_latvals)
    np.testing.assert_array_equal(lon_coord.points, expected_lonvals)

    assert lat_coord.has_bounds()
    assert lon_coord.has_bounds()


@pytest.mark.parametrize('spec', FAILING_SPECS)
def test_extract_regional_grid_failing(spec):
    """Test failing input for spec."""
    global_cube = _global_stock_cube('10x10')
    scheme = 'linear'

    with pytest.raises(ValueError):
        _ = regrid(global_cube, target_grid=spec, scheme=scheme)


@pytest.mark.parametrize('spec', PASSING_SPECS)
def test_spec_to_latlonvals(spec):
    """Test lat/lon val specification."""
    latvals, lonvals = _spec_to_latlonvals(**spec)

    lat_step = spec['step_latitude']
    assert latvals[0] == spec['start_latitude']
    lat_diff = latvals[-1] - latvals[0]
    assert Decimal(lat_diff) % Decimal(str(lat_step)) == 0
    np.testing.assert_allclose(np.diff(latvals), lat_step)
    assert spec['end_latitude'] >= latvals[-1]

    lon_step = spec['step_longitude']
    assert lonvals[0] == spec['start_longitude']
    lon_diff = lonvals[-1] - lonvals[0]
    assert Decimal(lon_diff) % Decimal(str(lon_step)) == 0
    np.testing.assert_allclose(np.diff(lonvals), lon_step)
    assert spec['end_longitude'] >= lonvals[-1]
