"""Test using sample data for :func:`esmvalcore.preprocessor._multimodel`."""

import pickle
import platform
from itertools import groupby
from pathlib import Path
from typing import Optional

import cf_units
import iris
import numpy as np
import pytest
from iris.coords import AuxCoord

from esmvalcore.preprocessor import extract_time
from esmvalcore.preprocessor._multimodel import multi_model_statistics

esmvaltool_sample_data = pytest.importorskip("esmvaltool_sample_data")

# Increase this number anytime you change the cached input data to the tests.
TEST_REVISION = 1

SPAN_PARAMS = ('overlap', 'full')


def assert_array_almost_equal(this, other, rtol=1e-7):
    """Assert that array `this` almost equals array `other`."""
    if np.ma.isMaskedArray(this) or np.ma.isMaskedArray(other):
        np.testing.assert_array_equal(this.mask, other.mask)

    np.testing.assert_allclose(this, other, rtol=rtol)


def assert_coords_equal(this: list, other: list):
    """Assert coords list `this` equals coords list `other`."""
    for this_coord, other_coord in zip(this, other):
        np.testing.assert_equal(this_coord.points, other_coord.points)
        assert this_coord.var_name == other_coord.var_name
        assert this_coord.standard_name == other_coord.standard_name
        assert this_coord.units == other_coord.units


def assert_metadata_equal(this, other):
    """Assert metadata `this` are equal to metadata `other`."""
    assert this.standard_name == other.standard_name
    assert this.long_name == other.long_name
    assert this.var_name == other.var_name
    assert this.units == other.units


def fix_metadata(cubes):
    """Fix metadata."""
    for cube in cubes:
        cube.coord('air_pressure').bounds = None


def preprocess_data(cubes, time_slice: Optional[dict] = None):
    """Regrid the data to the first cube and optional time-slicing."""
    # Increase TEST_REVISION anytime you make changes to this function.
    if time_slice:
        cubes = [extract_time(cube, **time_slice) for cube in cubes]

    first_cube = cubes[0]

    # regrid to first cube
    regrid_kwargs = {
        'grid': first_cube,
        'scheme': iris.analysis.Nearest(),
    }

    cubes = [cube.regrid(**regrid_kwargs) for cube in cubes]

    return cubes


def get_cache_key(value):
    """Get a cache key that is hopefully unique enough for unpickling.

    If this doesn't avoid problems with unpickling the cached data,
    manually clean the pytest cache with the command `pytest --cache-
    clear`.
    """
    py_version = platform.python_version()
    return (f'{value}_iris-{iris.__version__}_'
            f'numpy-{np.__version__}_python-{py_version}'
            f'rev-{TEST_REVISION}')


@pytest.fixture(scope="module")
def timeseries_cubes_month(request):
    """Load representative timeseries data."""
    # cache the cubes to save about 30-60 seconds on repeat use
    cache_key = get_cache_key("sample_data/monthly")
    data = request.config.cache.get(cache_key, None)

    if data:
        cubes = pickle.loads(data.encode('latin1'))
    else:
        # Increase TEST_REVISION anytime you make changes here.
        time_slice = {
            'start_year': 1985,
            'end_year': 1987,
            'start_month': 12,
            'end_month': 2,
            'start_day': 1,
            'end_day': 1,
        }
        cubes = esmvaltool_sample_data.load_timeseries_cubes(mip_table='Amon')
        cubes = preprocess_data(cubes, time_slice=time_slice)

        # cubes are not serializable via json, so we must go via pickle
        request.config.cache.set(cache_key,
                                 pickle.dumps(cubes).decode('latin1'))

    fix_metadata(cubes)

    return cubes


@pytest.fixture(scope="module")
def timeseries_cubes_day(request):
    """Load representative timeseries data grouped by calendar."""
    # cache the cubes to save about 30-60 seconds on repeat use
    cache_key = get_cache_key("sample_data/daily")
    data = request.config.cache.get(cache_key, None)

    if data:
        cubes = pickle.loads(data.encode('latin1'))

    else:
        # Increase TEST_REVISION anytime you make changes here.
        time_slice = {
            'start_year': 2001,
            'end_year': 2002,
            'start_month': 12,
            'end_month': 2,
            'start_day': 1,
            'end_day': 1,
        }
        cubes = esmvaltool_sample_data.load_timeseries_cubes(mip_table='day')
        cubes = preprocess_data(cubes, time_slice=time_slice)

        # cubes are not serializable via json, so we must go via pickle
        request.config.cache.set(cache_key,
                                 pickle.dumps(cubes).decode('latin1'))

    fix_metadata(cubes)

    def calendar(cube):
        return cube.coord('time').units.calendar

    # groupby requires sorted list
    grouped = groupby(sorted(cubes, key=calendar), key=calendar)

    cube_dict = {key: list(group) for key, group in grouped}

    return cube_dict


def multimodel_test(cubes, statistic, span, **kwargs):
    """Run multimodel test with some simple checks."""
    statistics = [statistic]

    result = multi_model_statistics(products=cubes,
                                    statistics=statistics,
                                    span=span,
                                    **kwargs)
    assert isinstance(result, dict)
    assert statistic in result

    return result


def multimodel_regression_test(cubes, span, name):
    """Run multimodel regression test.

    This test will fail if the input data or multimodel code changed. To
    update the data for the regression test, remove the corresponding
    `.nc` files in this directory and re-run the tests. The tests will
    fail the first time with a RuntimeError, because the reference data
    are being written.
    """
    statistic = 'mean'
    result = multimodel_test(cubes, statistic=statistic, span=span)
    result_cube = result[statistic]

    filename = Path(__file__).with_name(f'{name}-{span}-{statistic}.nc')
    if filename.exists():
        reference_cube = iris.load_cube(str(filename))

        assert_array_almost_equal(result_cube.data, reference_cube.data, 5e-7)
        assert_metadata_equal(result_cube.metadata, reference_cube.metadata)
        assert_coords_equal(result_cube.coords(), reference_cube.coords())

    else:
        # The test will fail if no regression data are available.
        iris.save(result_cube, filename)
        raise RuntimeError(f'Wrote reference data to {filename.absolute()}')


@pytest.mark.use_sample_data
@pytest.mark.parametrize('span', SPAN_PARAMS)
def test_multimodel_regression_month(timeseries_cubes_month, span):
    """Test statistic fail due to differing input coordinates (pressure).

    See https://github.com/ESMValGroup/ESMValCore/issues/956.

    """
    cubes = timeseries_cubes_month
    name = 'timeseries_monthly'
    msg = (
        "Multi-model statistics failed to merge input cubes into a single "
        "array"
    )
    with pytest.raises(ValueError, match=msg):
        multimodel_regression_test(name=name, span=span, cubes=cubes)


@pytest.mark.use_sample_data
@pytest.mark.parametrize('span', SPAN_PARAMS)
def test_multimodel_regression_day_standard(timeseries_cubes_day, span):
    """Test statistic."""
    calendar = 'standard' if cf_units.__version__ >= '3.1' else 'gregorian'
    cubes = timeseries_cubes_day[calendar]
    name = f'timeseries_daily_{calendar}'
    multimodel_regression_test(name=name, span=span, cubes=cubes)


@pytest.mark.use_sample_data
@pytest.mark.parametrize('span', SPAN_PARAMS)
def test_multimodel_regression_day_365_day(timeseries_cubes_day, span):
    """Test statistic."""
    calendar = '365_day'
    cubes = timeseries_cubes_day[calendar]
    name = f'timeseries_daily_{calendar}'
    multimodel_regression_test(name=name, span=span, cubes=cubes)


@pytest.mark.skip(
    reason='Cannot calculate statistics with single cube in list'
)
@pytest.mark.use_sample_data
@pytest.mark.parametrize('span', SPAN_PARAMS)
def test_multimodel_regression_day_360_day(timeseries_cubes_day, span):
    """Test statistic."""
    calendar = '360_day'
    cubes = timeseries_cubes_day[calendar]
    name = f'timeseries_daily_{calendar}'
    multimodel_regression_test(name=name, span=span, cubes=cubes)


@pytest.mark.skip(
    reason='Cannot calculate statistics with single cube in list'
)
@pytest.mark.use_sample_data
@pytest.mark.parametrize('span', SPAN_PARAMS)
def test_multimodel_regression_day_julian(timeseries_cubes_day, span):
    """Test statistic."""
    calendar = 'julian'
    cubes = timeseries_cubes_day[calendar]
    name = f'timeseries_daily_{calendar}'
    multimodel_regression_test(name=name, span=span, cubes=cubes)


@pytest.mark.use_sample_data
@pytest.mark.parametrize('span', SPAN_PARAMS)
def test_multimodel_regression_day_proleptic_gregorian(
    timeseries_cubes_day,
    span,
):
    """Test statistic."""
    calendar = 'proleptic_gregorian'
    cubes = timeseries_cubes_day[calendar]
    name = f'timeseries_daily_{calendar}'
    msg = (
        "Multi-model statistics failed to merge input cubes into a single "
        "array"
    )
    with pytest.raises(ValueError, match=msg):
        multimodel_regression_test(name=name, span=span, cubes=cubes)


@pytest.mark.use_sample_data
@pytest.mark.parametrize('span', SPAN_PARAMS)
def test_multimodel_no_vertical_dimension(timeseries_cubes_month, span):
    """Test statistic without vertical dimension using monthly data."""
    cubes = [cube[:, 0] for cube in timeseries_cubes_month]
    multimodel_test(cubes, span=span, statistic='mean')


@pytest.mark.use_sample_data
@pytest.mark.parametrize('span', SPAN_PARAMS)
def test_multimodel_merge_error(timeseries_cubes_month, span):
    """Test statistic with slightly different vertical coordinates.

    See https://github.com/ESMValGroup/ESMValCore/issues/956.

    """
    cubes = timeseries_cubes_month
    msg = (
        "Multi-model statistics failed to merge input cubes into a single "
        "array"
    )
    with pytest.raises(ValueError, match=msg):
        multimodel_test(cubes, span=span, statistic='mean')


@pytest.mark.use_sample_data
@pytest.mark.parametrize('span', SPAN_PARAMS)
def test_multimodel_only_time_dimension(timeseries_cubes_month, span):
    """Test statistic without only the time dimension using monthly data."""
    cubes = [cube[:, 0, 0, 0] for cube in timeseries_cubes_month]
    multimodel_test(cubes, span=span, statistic='mean')


@pytest.mark.use_sample_data
@pytest.mark.parametrize('span', SPAN_PARAMS)
def test_multimodel_no_time_dimension(timeseries_cubes_month, span):
    """Test statistic without time dimension using monthly data.

    Note: we collapse the air_pressure dimension here (by selecting only its
    first value) since the original coordinate differs slightly across cubes
    and leads to merge errors. See also
    https://github.com/ESMValGroup/ESMValCore/issues/956.

    """
    cubes = [cube[0, 0] for cube in timeseries_cubes_month]

    result = multimodel_test(cubes, span=span, statistic='mean')['mean']
    assert result.shape == (3, 2)


@pytest.mark.use_sample_data
@pytest.mark.parametrize('span', SPAN_PARAMS)
def test_multimodel_scalar_cubes(timeseries_cubes_month, span):
    """Test statistic with scalar cubes."""
    cubes = [cube[0, 0, 0, 0] for cube in timeseries_cubes_month]

    result = multimodel_test(cubes, span=span, statistic='mean')['mean']
    assert result.shape == ()
    assert result.coord('time').bounds is None


@pytest.mark.use_sample_data
@pytest.mark.parametrize('span', SPAN_PARAMS)
def test_multimodel_0d_1d_time_no_ignore_scalars(timeseries_cubes_month, span):
    """Test statistic fail on 0D and 1D time dimension using monthly data.

    Note: we collapse the air_pressure dimension here (by selecting only its
    first value) since the original coordinate differs slightly across cubes
    and leads to merge errors. See also
    https://github.com/ESMValGroup/ESMValCore/issues/956.

    """
    cubes = [cube[:, 0] for cube in timeseries_cubes_month]  # remove Z-dim
    cubes[1] = cubes[1][0]  # use 0D time dim for one cube

    msg = "Tried to align cubes in multi-model statistics, but failed for cube"
    with pytest.raises(ValueError, match=msg):
        multimodel_test(cubes, span=span, statistic='mean')


@pytest.mark.use_sample_data
@pytest.mark.parametrize('span', SPAN_PARAMS)
def test_multimodel_0d_1d_time_ignore_scalars(timeseries_cubes_month, span):
    """Test statistic fail on 0D and 1D time dimension using monthly data.

    Note: we collapse the air_pressure dimension here (by selecting only its
    first value) since the original coordinate differs slightly across cubes
    and leads to merge errors. See also
    https://github.com/ESMValGroup/ESMValCore/issues/956.

    """
    cubes = [cube[:, 0] for cube in timeseries_cubes_month]  # remove Z-dim
    cubes[1] = cubes[1][0]  # use 0D time dim for one cube

    msg = (
        "Multi-model statistics failed to merge input cubes into a single "
        "array: some cubes have a 'time' dimension, some do not have a 'time' "
        "dimension."
    )
    with pytest.raises(ValueError, match=msg):
        multimodel_test(
            cubes, span=span, statistic='mean', ignore_scalar_coords=True
        )


@pytest.mark.use_sample_data
@pytest.mark.parametrize('span', SPAN_PARAMS)
def test_multimodel_only_some_time_dimensions(timeseries_cubes_month, span):
    """Test statistic fail if only some cubes have time dimension.

    Note: we collapse the air_pressure dimension here (by selecting only its
    first value) since the original coordinate differs slightly across cubes
    and leads to merge errors. See also
    https://github.com/ESMValGroup/ESMValCore/issues/956.

    """
    cubes = [cube[:, 0] for cube in timeseries_cubes_month]  # remove Z-dim

    # Remove time dimension for one cube
    cubes[1] = cubes[1][0]
    cubes[1].remove_coord('time')

    msg = (
        "Multi-model statistics failed to merge input cubes into a single "
        "array: some cubes have a 'time' dimension, some do not have a 'time' "
        "dimension."
    )
    with pytest.raises(ValueError, match=msg):
        multimodel_test(cubes, span=span, statistic='mean')


@pytest.mark.use_sample_data
@pytest.mark.parametrize('span', SPAN_PARAMS)
def test_multimodel_diff_scalar_time_fail(timeseries_cubes_month, span):
    """Test statistic fail on different scalar time dimensions.

    Note: we collapse the air_pressure dimension here (by selecting only its
    first value) since the original coordinate differs slightly across cubes
    and leads to merge errors. See also
    https://github.com/ESMValGroup/ESMValCore/issues/956.

    """
    cubes = [cube[0, 0] for cube in timeseries_cubes_month]

    # Use different scalar time point and bounds for one cube
    cubes[1].coord('time').points = 20.0
    cubes[1].coord('time').bounds = [0.0, 40.0]

    msg = "Tried to align cubes in multi-model statistics, but failed for cube"
    with pytest.raises(ValueError, match=msg):
        multimodel_test(cubes, span=span, statistic='mean')


@pytest.mark.use_sample_data
@pytest.mark.parametrize('span', SPAN_PARAMS)
def test_multimodel_diff_scalar_time_ignore(timeseries_cubes_month, span):
    """Ignore different scalar time dimensions.

    Note: we collapse the air_pressure dimension here (by selecting only its
    first value) since the original coordinate differs slightly across cubes
    and leads to merge errors. See also
    https://github.com/ESMValGroup/ESMValCore/issues/956.

    """
    cubes = [cube[0, 0] for cube in timeseries_cubes_month]

    # Use different scalar time point and bounds for one cube
    cubes[1].coord('time').points = 20.0
    cubes[1].coord('time').bounds = [0.0, 40.0]

    result = multimodel_test(
        cubes, span=span, statistic='mean', ignore_scalar_coords=True
    )['mean']
    assert result.shape == (3, 2)


@pytest.mark.use_sample_data
@pytest.mark.parametrize('span', SPAN_PARAMS)
def test_multimodel_ignore_scalar_coords(timeseries_cubes_month, span):
    """Test statistic does not fail on different scalar coords when ignored.

    Note: we collapse the air_pressure dimension here (by selecting only its
    first value) since the original coordinate differs slightly across cubes
    and leads to merge errors. See also
    https://github.com/ESMValGroup/ESMValCore/issues/956.

    """
    cubes = [cube[0, 0] for cube in timeseries_cubes_month]
    for (idx, cube) in enumerate(cubes):
        aux_coord = AuxCoord(0.0, var_name=f'name_{idx}')
        cube.add_aux_coord(aux_coord, ())

    result = multimodel_test(
        cubes, span=span, statistic='mean', ignore_scalar_coords=True
    )['mean']
    assert result.shape == (3, 2)

    # Make sure that the input cubes still contain the scalar coords
    for (idx, cube) in enumerate(cubes):
        assert cube.coord(var_name=f'name_{idx}', dimensions=())


@pytest.mark.use_sample_data
@pytest.mark.parametrize('span', SPAN_PARAMS)
def test_multimodel_do_not_ignore_scalar_coords(timeseries_cubes_month, span):
    """Test statistic fail on different scalar coords.

    Note: we collapse the air_pressure dimension here (by selecting only its
    first value) since the original coordinate differs slightly across cubes
    and leads to merge errors. See also
    https://github.com/ESMValGroup/ESMValCore/issues/956.

    """
    cubes = [cube[0, 0] for cube in timeseries_cubes_month]
    for (idx, cube) in enumerate(cubes):
        aux_coord = AuxCoord(0.0, var_name=f'name_{idx}')
        cube.add_aux_coord(aux_coord, ())

    msg = (
        "Multi-model statistics failed to merge input cubes into a single "
        "array"
    )
    with pytest.raises(ValueError, match=msg):
        multimodel_test(cubes, span=span, statistic='mean')
