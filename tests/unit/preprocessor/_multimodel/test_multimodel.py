"""Unit test for :func:`esmvalcore.preprocessor._multimodel`."""

from datetime import datetime
from unittest import mock

import cftime
import dask.array as da
import iris
import iris.coord_categorisation
import numpy as np
import pytest
from cf_units import Unit
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube, CubeList

import esmvalcore.preprocessor._multimodel as mm
from esmvalcore.iris_helpers import date2num
from esmvalcore.preprocessor import multi_model_statistics
from esmvalcore.preprocessor._supplementary_vars import add_ancillary_variable

SPAN_OPTIONS = ('overlap', 'full')

FREQUENCY_OPTIONS = ('daily', 'monthly', 'yearly')  # hourly

CALENDAR_OPTIONS = ('360_day', '365_day', 'standard', 'proleptic_gregorian',
                    'julian')

EQUAL_NAMES = [
    ['var_name'],
    ['standard_name'],
    ['long_name'],
    ['var_name', 'standard_name'],
    ['var_name', 'long_name'],
    ['standard_name', 'long_name'],
    ['var_name', 'standard_name', 'long_name'],
]


def assert_array_allclose(this, other):
    """Assert that array `this` is close to array `other`."""
    if np.ma.isMaskedArray(this) or np.ma.isMaskedArray(other):
        np.testing.assert_array_equal(this.mask, other.mask)

    np.testing.assert_allclose(this, other)


@pytest.fixture
def cubes_with_arbitrary_dimensions():
    """Create cubes with non-standard dimensions."""
    a_coord = DimCoord([1, 2, 3], var_name='a')
    b_coord = DimCoord([1], var_name='b')
    s_coord = AuxCoord(0, var_name='s')
    cube_kwargs = {
        'var_name': 'x',
        'dim_coords_and_dims': [(a_coord, 0), (b_coord, 1)],
        'aux_coords_and_dims': [(s_coord, ())],
    }

    cubes = CubeList([
        Cube([[0.0], [0.0], [0.0]], **cube_kwargs),
        Cube([[0.0], [2.0], [1.0]], **cube_kwargs),
        Cube([[0.0], [4.0], [2.0]], **cube_kwargs),
    ])

    return cubes


@pytest.fixture
def cubes_5d():
    """Create 5d cubes."""
    a_coord = DimCoord([1], var_name='a')
    b_coord = DimCoord([1], var_name='b')
    c_coord = DimCoord([1], var_name='c')
    d_coord = DimCoord([1], var_name='d')
    e_coord = DimCoord([1], var_name='e')
    coord_spec = [
        (a_coord, 0),
        (b_coord, 1),
        (c_coord, 2),
        (d_coord, 3),
        (e_coord, 4),
    ]

    cubes = CubeList([
        Cube(np.full((1, 1, 1, 1, 1), 1.0), dim_coords_and_dims=coord_spec),
        Cube(np.full((1, 1, 1, 1, 1), 2.0), dim_coords_and_dims=coord_spec),
    ])

    return cubes


def timecoord(frequency,
              calendar='standard',
              offset='days since 1850-01-01',
              num=3):
    """Return a time coordinate with the given time points and calendar."""
    time_points = range(1, num + 1)

    if frequency == 'hourly':
        dates = [datetime(1850, 1, 1, i, 0, 0) for i in time_points]
    if frequency == 'daily':
        dates = [datetime(1850, 1, i, 0, 0, 0) for i in time_points]
    elif frequency == 'monthly':
        dates = [datetime(1850, i, 15, 0, 0, 0) for i in time_points]
    elif frequency == 'yearly':
        dates = [datetime(1850 + i - 1, 7, 1, 0, 0, 0) for i in time_points]

    unit = Unit(offset, calendar=calendar)
    points = date2num(dates, unit)
    return DimCoord(points, standard_name='time', units=unit)


def generate_cube_from_dates(
    dates,
    calendar='standard',
    offset='days since 1850-01-01',
    fill_val=1,
    len_data=3,
    var_name=None,
    lazy=False,
):
    """Generate test cube from list of dates / frequency specification.

    Parameters
    ----------
    calendar : str or list
        Date frequency: 'hourly' / 'daily' / 'monthly' / 'yearly' or
        list of datetimes.
    offset : str
        Offset to use
    fill_val : int
        Value to fill the data with
    len_data : int
        Number of data / time points
    var_name : str
        Name of the data variable

    Returns
    -------
    iris.cube.Cube
    """
    if isinstance(dates, str):
        time = timecoord(dates, calendar, offset=offset, num=len_data)
    else:
        len_data = len(dates)
        unit = Unit(offset, calendar=calendar)
        time = DimCoord(date2num(dates, unit),
                        standard_name='time',
                        units=unit)

    data = np.array((fill_val, ) * len_data, dtype=np.float32)

    if lazy:
        data = da.from_array(data)

    return Cube(data, dim_coords_and_dims=[(time, 0)], var_name=var_name)


def get_cubes_for_validation_test(frequency, lazy=False):
    """Set up cubes used for testing multimodel statistics."""
    # Simple 1d cube with standard time cord
    cube1 = generate_cube_from_dates(frequency, lazy=lazy)

    # Cube with masked data
    cube2 = cube1.copy()
    data2 = np.ma.array([5, 5, 5], mask=[True, False, False], dtype=np.float32)
    if lazy:
        data2 = da.from_array(data2)
    cube2.data = data2

    # Cube with deviating time coord
    cube3 = generate_cube_from_dates(frequency,
                                     calendar='360_day',
                                     offset='days since 1950-01-01',
                                     len_data=2,
                                     fill_val=9,
                                     lazy=lazy)

    return [cube1, cube2, cube3]


def get_cube_for_equal_coords_test(num_cubes):
    """Set up cubes with equal auxiliary coordinates."""
    cubes = []

    for num in range(num_cubes):
        cube = generate_cube_from_dates('monthly')
        cubes.append(cube)

    # Create cubes that have one exactly equal coordinate ('year'), one
    # coordinate with matching names ('m') and one coordinate with non-matching
    # names
    year_coord = AuxCoord([1, 2, 3], var_name='year', long_name='year',
                          units='1', attributes={'test': 1})
    m_coord = AuxCoord([1, 2, 3], var_name='m', long_name='m', units='s',
                       attributes={'test': 0})
    x_coord = AuxCoord([1, 2, 3], var_name='x', long_name='x', units='s',
                       attributes={'test': 2})
    for (idx, cube) in enumerate(cubes):
        new_m_coord = m_coord.copy()
        new_m_coord.var_name = f'm_{idx}'
        new_x_coord = x_coord.copy()
        new_x_coord.long_name = f'x_{idx}'
        cube.add_aux_coord(year_coord.copy(), 0)
        cube.add_aux_coord(new_m_coord, 0)
        cube.add_aux_coord(new_x_coord, 0)
        assert cube.coord('year').metadata is not year_coord.metadata
        assert cube.coord('year').metadata == year_coord.metadata
        assert cube.coord('m').metadata is not m_coord.metadata
        assert cube.coord('m').metadata != m_coord.metadata
        assert cube.coord(f'x_{idx}').metadata is not x_coord.metadata
        assert cube.coord(f'x_{idx}').metadata != x_coord.metadata

    return cubes


VALIDATION_DATA_SUCCESS = (
    ('full', 'mean', (5, 5, 3)),
    ('full', {'operator': 'mean'}, (5, 5, 3)),
    ('full', 'std_dev', (5.656854249492381, 4, 2.8284271247461903)),
    ('full', 'std', (5.656854249492381, 4, 2.8284271247461903)),
    ('full', 'min', (1, 1, 1)),
    ('full', 'max', (9, 9, 5)),
    ('full', 'median', (5, 5, 3)),
    ('full', {'operator': 'percentile', 'percent': 50.0}, (5, 5, 3)),
    ('full', 'p50', (5, 5, 3)),
    ('full', 'p99.5', (8.96, 8.96, 4.98)),
    ('full', 'peak', (9, 9, 5)),
    ('overlap', 'mean', (5, 5)),
    ('overlap', 'std_dev', (5.656854249492381, 4)),
    ('overlap', 'std', (5.656854249492381, 4)),
    ('overlap', 'min', (1, 1)),
    ('overlap', 'max', (9, 9)),
    ('overlap', 'median', (5, 5)),
    ('overlap', {'operator': 'percentile', 'percent': 50.0}, (5, 5)),
    ('overlap', 'p50', (5, 5)),
    ('overlap', 'p99.5', (8.96, 8.96)),
    ('overlap', 'peak', (9, 9)),
    # test multiple statistics
    ('overlap', ('min', 'max'), ((1, 1), (9, 9))),
    ('overlap', ('min', {'operator': 'max'}), ((1, 1), (9, 9))),
    ('full', ('min', 'max'), ((1, 1, 1), (9, 9, 5))),
    ('full', (
        {'operator': 'percentile', 'percent': 50.0},
        {'operator': 'percentile', 'percent': 99.5}
    ), ((5, 5, 3), (8.96, 8.96, 4.98))),
)


@pytest.mark.parametrize(
    'length,slices',
    [
        (1, [slice(0, 1)]),
        (25000, [slice(0, 8334),
                 slice(8334, 16668),
                 slice(16668, 25000)]),
    ],
)
def test_compute_slices(length, slices):
    """Test cube `_compute_slices`."""
    cubes = [
        Cube(da.empty([length, 50, 100], dtype=np.float32)) for _ in range(5)
    ]
    result = list(mm._compute_slices(cubes))
    assert result == slices


def test_compute_slices_exceed_end_index():
    """Test that ``_compute_slices`` terminates when exceeding end index."""
    # The following settings will result in a cube length of 71, 10 slices and
    # a slice length of 8. Thus, without early termination, the last slice
    # would be slice(72, 71), which would result in an exception.
    cube_data = mock.Mock(nbytes=1.1 * 2**30)  # roughly 1.1 GiB
    cube = mock.Mock(spec=Cube, data=cube_data, shape=(71,))
    cubes = [cube] * 9

    slices = list(mm._compute_slices(cubes))

    # Early termination lead to 9 (instead of 10) slices
    assert len(slices) == 9
    expected_slices = [
        slice(0, 8, None),
        slice(8, 16, None),
        slice(16, 24, None),
        slice(24, 32, None),
        slice(32, 40, None),
        slice(40, 48, None),
        slice(48, 56, None),
        slice(56, 64, None),
        slice(64, 71, None),
    ]
    assert slices == expected_slices


def test_compute_slices_equals_end_index():
    """Test that ``_compute_slices`` terminates when reaching end index."""
    # The following settings will result in a cube length of 36, 13 slices and
    # a slice length of 3. Thus, without early termination, the last slice
    # would be slice(36, 39), which would result in an exception.
    cube_data = mock.Mock(nbytes=1.05 * 2**30)  # roughly 1.05 GiB
    cube = mock.Mock(spec=Cube, data=cube_data, shape=(36,))
    cubes = [cube] * 12

    slices = list(mm._compute_slices(cubes))

    # Early termination lead to 12 (instead of 13) slices
    assert len(slices) == 12
    expected_slices = [
        slice(0, 3, None),
        slice(3, 6, None),
        slice(6, 9, None),
        slice(9, 12, None),
        slice(12, 15, None),
        slice(15, 18, None),
        slice(18, 21, None),
        slice(21, 24, None),
        slice(24, 27, None),
        slice(27, 30, None),
        slice(30, 33, None),
        slice(33, 36, None),
    ]
    assert slices == expected_slices


@pytest.mark.parametrize('frequency', FREQUENCY_OPTIONS)
@pytest.mark.parametrize('span, statistics, expected', VALIDATION_DATA_SUCCESS)
def test_multimodel_statistics(frequency, span, statistics, expected):
    """High level test for multicube statistics function."""
    cubes = get_cubes_for_validation_test(frequency)

    if isinstance(statistics, (str, dict)):
        statistics = (statistics, )
        expected = (expected, )

    result = multi_model_statistics(cubes, span, statistics)

    assert isinstance(result, dict)
    stat_ids = [mm._get_stat_identifier(s) for s in statistics]
    assert set(result.keys()) == set(stat_ids)

    for i, stat_id in enumerate(stat_ids):
        result_cube = result[stat_id]
        # make sure that temporary coord has been removed
        with pytest.raises(iris.exceptions.CoordinateNotFoundError):
            result_cube.coord('multi-model')
        # test that real data in => real data out
        assert result_cube.has_lazy_data() is False
        expected_data = np.ma.array(expected[i], mask=False)
        assert_array_allclose(result_cube.data, expected_data)


@pytest.mark.parametrize('span', SPAN_OPTIONS)
def test_lazy_data_consistent_times(span):
    """Test laziness of multimodel statistics with consistent time axis."""
    cubes = (
        generate_cube_from_dates('monthly', fill_val=1, lazy=True),
        generate_cube_from_dates('monthly', fill_val=3, lazy=True),
        generate_cube_from_dates('monthly', fill_val=6, lazy=True),
    )

    for cube in cubes:
        assert cube.has_lazy_data()

    statistic = 'sum'
    statistics = (statistic, )

    result = mm._multicube_statistics(cubes, span=span, statistics=statistics)

    result_cube = result[statistic]
    assert result_cube.has_lazy_data()


@pytest.mark.parametrize('span', SPAN_OPTIONS)
def test_lazy_data_inconsistent_times(span):
    """Test laziness of multimodel statistics with inconsistent time axis.

    This hits `_align`, which adds additional computations which must be
    lazy.
    """
    cubes = (
        generate_cube_from_dates(
            [datetime(1850, i, 15, 0, 0, 0) for i in range(1, 10)], lazy=True),
        generate_cube_from_dates(
            [datetime(1850, i, 15, 0, 0, 0) for i in range(3, 8)], lazy=True),
        generate_cube_from_dates(
            [datetime(1850, i, 15, 0, 0, 0) for i in range(2, 9)], lazy=True),
    )

    for cube in cubes:
        assert cube.has_lazy_data()

    statistic = 'sum'
    statistics = (statistic, )

    result = mm._multicube_statistics(cubes, span=span, statistics=statistics)

    result_cube = result[statistic]
    assert result_cube.has_lazy_data()


@pytest.mark.parametrize('span', SPAN_OPTIONS)
def test_multicube_stats_dict_keys(span):
    """Test output dict keys of ``_multicube_statistics``."""
    cubes = (
        generate_cube_from_dates('monthly', fill_val=1),
        generate_cube_from_dates('monthly', fill_val=3),
        generate_cube_from_dates('monthly', fill_val=6),
    )
    statistics = [
        'mean',
        {'operator': 'sum'},
        {'operator': 'percentile', 'percent': 50},
        {'operator': 'percentile', 'percent': 95.0},
    ]

    result = mm._multicube_statistics(cubes, span=span, statistics=statistics)

    assert isinstance(result, dict)
    assert len(result) == 4
    assert 'mean' in result
    assert 'sum' in result
    assert 'percentile50' in result
    assert 'percentile95.0' in result


VALIDATION_DATA_FAIL = (
    ('percentile', ValueError),
    ('wpercentile', ValueError),
    ('count', ValueError),
    ('proportion', ValueError),
)


@pytest.mark.parametrize('statistic, error', VALIDATION_DATA_FAIL)
def test_unsupported_statistics_fail(statistic, error):
    """Check that unsupported statistics raise an exception."""
    cubes = get_cubes_for_validation_test('monthly')
    span = 'overlap'
    statistics = (statistic, )
    with pytest.raises(error):
        _ = multi_model_statistics(cubes, span, statistics)


@pytest.mark.parametrize('calendar1, calendar2, expected', (
    ('360_day', '360_day', ('360_day',)),
    ('365_day', '365_day', ('365_day',)),
    ('365_day', '360_day', ('standard', 'gregorian')),
    ('360_day', '365_day', ('standard', 'gregorian')),
    ('standard', '365_day', ('standard', 'gregorian')),
    ('proleptic_gregorian', 'julian', ('standard', 'gregorian')),
    ('julian', '365_day', ('standard', 'gregorian')),
))
def test_get_consistent_time_unit(calendar1, calendar2, expected):
    """Test same calendar returned or default if calendars differ.

    Expected behaviour: If the calendars are the same, return that one.
    If the calendars are not the same, return 'standard'.
    """
    cubes = (
        generate_cube_from_dates('monthly', calendar=calendar1),
        generate_cube_from_dates('monthly', calendar=calendar2),
    )

    result = mm._get_consistent_time_unit(cubes)
    assert result.calendar in expected


@pytest.mark.parametrize('span', SPAN_OPTIONS)
def test_align(span):
    """Test _align function."""
    # TODO --> check that if a cube is extended,
    #          the extended points are masked (not NaN!)
    len_data = 3

    cubes = []

    for calendar in CALENDAR_OPTIONS:
        cube = generate_cube_from_dates('monthly',
                                        calendar=calendar,
                                        len_data=3)
        cubes.append(cube)

    result_cubes = mm._align_time_coord(cubes, span)

    calendars = set(cube.coord('time').units.calendar for cube in result_cubes)

    assert len(calendars) == 1
    assert list(calendars)[0] in ('standard', 'gregorian')

    shapes = set(cube.shape for cube in result_cubes)

    assert len(shapes) == 1
    assert tuple(shapes)[0] == (len_data, )


@pytest.mark.parametrize('span', SPAN_OPTIONS)
def test_combine_same_shape(span):
    """Test _combine with same shape of cubes."""
    len_data = 3
    num_cubes = 5
    cubes = []

    for i in range(num_cubes):
        cube = generate_cube_from_dates('monthly',
                                        '360_day',
                                        fill_val=i,
                                        len_data=len_data)
        cubes.append(cube)

    result_cube = mm._combine(cubes)

    dim_coord = result_cube.coord(mm.CONCAT_DIM)
    assert dim_coord.var_name == mm.CONCAT_DIM
    assert result_cube.shape == (num_cubes, len_data)

    desired = np.linspace((0, ) * len_data,
                          num_cubes - 1,
                          num=num_cubes,
                          dtype=int)
    np.testing.assert_equal(result_cube.data, desired)


def test_combine_different_shape_fail():
    """Test _combine with inconsistent data."""
    num_cubes = 5
    cubes = []

    for num in range(1, num_cubes + 1):
        cube = generate_cube_from_dates('monthly', '360_day', len_data=num)
        cubes.append(cube)

    msg = (
        "Multi-model statistics failed to merge input cubes into a single "
        "array"
    )
    with pytest.raises(ValueError, match=msg):
        mm._combine(cubes)


def test_combine_inconsistent_var_names_fail():
    """Test _combine with inconsistent var names."""
    num_cubes = 5
    cubes = []

    for num in range(num_cubes):
        cube = generate_cube_from_dates('monthly',
                                        '360_day',
                                        var_name=f'test_var_{num}')
        cubes.append(cube)

    msg = (
        "Multi-model statistics failed to merge input cubes into a single "
        "array"
    )
    with pytest.raises(ValueError, match=msg):
        mm._combine(cubes)


def test_combine_differing_scalar_coords_fail():
    """Test _combine with differing scalar coordinates."""
    cubes = CubeList(generate_cube_from_dates('monthly') for _ in range(2))
    scalar_coord_0 = AuxCoord(0.0, standard_name='height', units='m')
    cubes[0].add_aux_coord(scalar_coord_0, ())

    msg = (
        "Multi-model statistics failed to merge input cubes into a single "
        "array"
    )
    with pytest.raises(ValueError, match=msg):
        mm._combine(cubes)


@pytest.mark.parametrize('scalar_coord', ['p0', 'ptop'])
def test_combine_with_special_scalar_coords_to_remove(scalar_coord):
    """Test _combine with scalar coordinates that should be removed."""
    cubes = CubeList(generate_cube_from_dates('monthly') for _ in range(5))
    scalar_coord_0 = AuxCoord(0.0, var_name=scalar_coord)
    scalar_coord_1 = AuxCoord(1.0, var_name=scalar_coord)
    cubes[0].add_aux_coord(scalar_coord_0, ())
    cubes[1].add_aux_coord(scalar_coord_1, ())

    merged_cube = mm._combine(cubes)
    assert merged_cube.shape == (5, 3)


def test_combine_equal_coordinates():
    """Test ``_combine`` with equal input coordinates."""
    cubes = get_cube_for_equal_coords_test(5)
    merged_cube = mm._combine(cubes)

    # The equal coordinate ('year') was not changed
    assert merged_cube.coord('year').var_name == 'year'
    assert merged_cube.coord('year').standard_name is None
    assert merged_cube.coord('year').long_name == 'year'
    assert merged_cube.coord('year').attributes == {'test': 1}


def test_combine_non_equal_coordinates():
    """Test ``_combine`` with non-equal input coordinates."""
    cubes = get_cube_for_equal_coords_test(5)
    merged_cube = mm._combine(cubes)

    # The var_name of the matching name coordinate ('m') has been removed, and
    # the non-equal one ('x') does not have a long_name anymore
    # Both coordinates lost their attributes
    assert merged_cube.coord('m').var_name is None
    assert merged_cube.coord('m').standard_name is None
    assert merged_cube.coord('m').long_name == 'm'
    assert merged_cube.coord('m').attributes == {}
    assert merged_cube.coord('x').var_name == 'x'
    assert merged_cube.coord('x').standard_name is None
    assert merged_cube.coord('x').long_name is None
    assert merged_cube.coord('x').attributes == {}


def test_equalise_coordinate_metadata_no_cubes():
    """Test _equalise_coordinate_metadata doesn't fail with empty cubes."""
    mm._equalise_coordinate_metadata([])


def test_equalise_coordinate_metadata_one_cube():
    """Test _equalise_coordinate_metadata doesn't fail with a single cubes."""
    cube = generate_cube_from_dates('monthly')
    new_cube = cube.copy()
    mm._equalise_coordinate_metadata([new_cube])
    assert new_cube is not cube
    assert new_cube == cube


@pytest.mark.parametrize('span', SPAN_OPTIONS)
def test_edge_case_different_time_offsets(span):
    cubes = (
        generate_cube_from_dates('monthly',
                                 '360_day',
                                 offset='days since 1888-01-01'),
        generate_cube_from_dates('monthly',
                                 '360_day',
                                 offset='days since 1899-01-01'),
    )

    statistic = 'min'
    statistics = (statistic, )

    result = multi_model_statistics(cubes, span, statistics)

    result_cube = result[statistic]

    time_coord = result_cube.coord('time')

    assert time_coord.units.calendar in ('standard', 'gregorian')
    assert time_coord.units.origin == 'days since 1850-01-01'

    desired = np.array((14., 45., 73.))
    np.testing.assert_array_equal(time_coord.points, desired)


def generate_cubes_with_non_overlapping_timecoords():
    """Generate sample data where time coords do not overlap."""
    time_points = range(1, 4)
    dates1 = [datetime(1850, i, 15, 0, 0, 0) for i in time_points]
    dates2 = [datetime(1950, i, 15, 0, 0, 0) for i in time_points]

    return (
        generate_cube_from_dates(dates1),
        generate_cube_from_dates(dates2),
    )


@pytest.mark.xfail(reason='Multimodel statistics returns the original cubes.')
def test_edge_case_time_no_overlap_fail():
    """Test case when time coords do not overlap using span='overlap'.

    Expected behaviour: `multi_model_statistics` should fail if time
    points are not overlapping.
    """
    cubes = generate_cubes_with_non_overlapping_timecoords()

    statistic = 'min'
    statistics = (statistic, )

    with pytest.raises(ValueError):
        _ = multi_model_statistics(cubes, 'overlap', statistics)


def test_edge_case_time_no_overlap_success():
    """Test case when time coords do not overlap using span='full'.

    Expected behaviour: `multi_model_statistics` should use all
    available time points.
    """
    cubes = generate_cubes_with_non_overlapping_timecoords()

    statistic = 'min'
    statistics = (statistic, )

    result = multi_model_statistics(cubes, 'full', statistics)
    result_cube = result[statistic]

    assert result_cube.coord('time').shape == (6, )


@pytest.mark.parametrize('span', SPAN_OPTIONS)
def test_edge_case_time_not_in_middle_of_months(span):
    """Test case when time coords are not on 15th for monthly data.

    Expected behaviour: `multi_model_statistics` will set all dates to
    the 15th.
    """
    time_points = range(1, 4)
    dates1 = [datetime(1850, i, 12, 0, 0, 0) for i in time_points]
    dates2 = [datetime(1850, i, 25, 0, 0, 0) for i in time_points]

    cubes = (
        generate_cube_from_dates(dates1),
        generate_cube_from_dates(dates2),
    )

    statistic = 'min'
    statistics = (statistic, )

    result = multi_model_statistics(cubes, span, statistics)
    result_cube = result[statistic]

    time_coord = result_cube.coord('time')

    desired = np.array((14., 45., 73.))
    np.testing.assert_array_equal(time_coord.points, desired)


@pytest.mark.parametrize('span', SPAN_OPTIONS)
def test_edge_case_sub_daily_data_fail(span):
    """Test case when cubes with sub-daily time coords are passed."""
    cube = generate_cube_from_dates('hourly')
    cubes = (cube, cube)

    statistic = 'min'
    statistics = (statistic, )

    with pytest.raises(ValueError):
        _ = multi_model_statistics(cubes, span, statistics)


def test_unify_time_coordinates():
    """Test set common calendar."""
    cube1 = generate_cube_from_dates('monthly',
                                     calendar='360_day',
                                     offset='days since 1850-01-01')
    cube2 = generate_cube_from_dates('monthly',
                                     calendar='standard',
                                     offset='days since 1943-05-16')

    mm._unify_time_coordinates([cube1, cube2])

    assert cube1.coord('time') == cube2.coord('time')


class PreprocessorFile:
    """Mockup to test output of multimodel."""

    def __init__(self, cube=None, attributes=None):
        if cube:
            self.cubes = [cube]
        if attributes:
            self.attributes = attributes

    def wasderivedfrom(self, product):
        pass

    def group(self, keys: list) -> str:
        """Generate group keyword.

        Returns a string that identifies a group. Concatenates a list of
        values from .attributes
        """
        if not keys:
            return ''

        if isinstance(keys, str):
            keys = [keys]

        identifier = []
        for key in keys:
            attribute = self.attributes.get(key)
            if attribute:
                if isinstance(attribute, (list, tuple)):
                    attribute = '-'.join(attribute)
                identifier.append(attribute)

        return '_'.join(identifier)


def test_return_products():
    """Check that the right product set is returned."""
    cube1 = generate_cube_from_dates('monthly', fill_val=1)
    cube2 = generate_cube_from_dates('monthly', fill_val=9)

    input1 = PreprocessorFile(cube1)
    input2 = PreprocessorFile(cube2)

    products = set([input1, input2])

    output = PreprocessorFile()
    output_products = {'': {'mean': output}}

    kwargs = {
        'statistics': ['mean'],
        'span': 'full',
        'output_products': output_products['']
    }

    result1 = mm._multiproduct_statistics(products,
                                          keep_input_datasets=True,
                                          **kwargs)

    result2 = mm._multiproduct_statistics(products,
                                          keep_input_datasets=False,
                                          **kwargs)

    assert result1 == set([input1, input2, output])
    assert result2 == set([output])

    kwargs['output_products'] = output_products
    result3 = mm.multi_model_statistics(products, **kwargs)
    result4 = mm.multi_model_statistics(products,
                                        keep_input_datasets=False,
                                        **kwargs)

    assert result3 == result1
    assert result4 == result2


def test_ensemble_products():
    cube1 = generate_cube_from_dates('monthly', fill_val=1)
    cube2 = generate_cube_from_dates('monthly', fill_val=9)

    attributes1 = {
        'project': 'project', 'dataset': 'dataset',
        'exp': 'exp', 'ensemble': '1'}
    input1 = PreprocessorFile(cube1, attributes=attributes1)

    attributes2 = {
        'project': 'project', 'dataset': 'dataset',
        'exp': 'exp', 'ensemble': '2'}
    input2 = PreprocessorFile(cube2, attributes=attributes2)

    attributes3 = {
        'project': 'project', 'dataset': 'dataset2',
        'exp': 'exp', 'ensemble': '1'}
    input3 = PreprocessorFile(cube1, attributes=attributes3)

    attributes4 = {
        'project': 'project', 'dataset': 'dataset2',
        'exp': 'exp', 'ensemble': '2'}

    input4 = PreprocessorFile(cube1, attributes=attributes4)
    products = set([input1, input2, input3, input4])

    output1 = PreprocessorFile()
    output2 = PreprocessorFile()
    output_products = {
        'project_dataset_exp': {'mean': output1},
        'project_dataset2_exp': {'mean': output2}}

    kwargs = {
        'statistics': ['mean'],
        'output_products': output_products,
    }

    result = mm.ensemble_statistics(
        products, **kwargs)
    assert len(result) == 2


def test_ignore_tas_scalar_height_coord():
    """Ignore conflicting aux_coords for height in tas."""
    tas_2m = generate_cube_from_dates("monthly")
    tas_1p5m = generate_cube_from_dates("monthly")

    for cube, height in zip([tas_2m, tas_1p5m], [2., 1.5]):
        cube.rename("air_temperature")
        cube.attributes["short_name"] = "tas"
        cube.add_aux_coord(
            iris.coords.AuxCoord([height], var_name="height", units="m"))

    result = mm.multi_model_statistics(
        [tas_2m, tas_2m.copy(), tas_1p5m], statistics=['mean'], span='full')

    # iris automatically averages the value of the scalar coordinate.
    assert len(result['mean'].coords("height")) == 1
    assert result["mean"].coord("height").points == 1.75


PRODUCTS = [
    CubeList(generate_cube_from_dates('monthly') for _ in range(3)),
    [
        PreprocessorFile(generate_cube_from_dates('monthly')) for _ in range(3)
    ],
]
SCALAR_COORD = AuxCoord(2.0, standard_name='height', units='m')
PRODUCTS[0][0].add_aux_coord(SCALAR_COORD, ())
PRODUCTS[1][0].cubes[0].add_aux_coord(SCALAR_COORD, ())
PRODUCTS[1] = set(PRODUCTS[1])


@pytest.mark.parametrize('products', PRODUCTS)
def test_ignore_different_scalar_coords(products):
    """Ignore different scalar coords if desired."""
    stat = 'mean'
    output = PreprocessorFile()
    output_products = {'': {stat: output}}
    kwargs = {
        'statistics': [stat],
        'span': 'full',
        'output_products': output_products,
        'keep_input_datasets': False,
        'ignore_scalar_coords': True,
    }

    results = mm.multi_model_statistics(products, **kwargs)

    assert len(results) == 1
    if isinstance(results, dict):  # for cube as input
        cube = results[stat]
    else:  # for PreprocessorFile as input
        result = next(iter(results))
        assert len(result.cubes) == 1
        cube = result.cubes[0]
    assert not cube.coords(dimensions=())


@pytest.mark.parametrize('products', PRODUCTS)
def test_do_not_ignore_different_scalar_coords(products):
    """Do not ignore different scalar coords if desired."""
    stat = 'mean'
    output = PreprocessorFile()
    output_products = {'': {stat: output}}
    kwargs = {
        'statistics': [stat],
        'span': 'full',
        'output_products': output_products,
        'keep_input_datasets': False,
    }

    msg = (
        "Multi-model statistics failed to merge input cubes into a single "
        "array"
    )
    with pytest.raises(ValueError, match=msg):
        mm.multi_model_statistics(products, **kwargs)


def test_daily_inconsistent_calendars():
    """Determine behaviour for inconsistent calendars.

    Deviating calendars should be converted to standard. Missing data
    inside original bounds is filled with nearest neighbour Missing data
    outside original bounds is masked.
    """
    ref_standard = Unit("days since 1850-01-01", calendar="standard")
    ref_noleap = Unit("days since 1850-01-01", calendar="noleap")
    start = date2num(datetime(1852, 1, 1), ref_standard)

    # 1852 is a leap year, and include 1 extra day at the end
    leapdates = cftime.num2date(start + np.arange(367),
                                ref_standard.name, ref_standard.calendar)

    noleapdates = cftime.num2date(start + np.arange(365),
                                  ref_noleap.name, ref_noleap.calendar)

    leapcube = generate_cube_from_dates(
        leapdates,
        calendar='standard',
        offset='days since 1850-01-01',
        fill_val=1,
    )

    noleapcube = generate_cube_from_dates(
        noleapdates,
        calendar='noleap',
        offset='days since 1850-01-01',
        fill_val=3,
    )

    cubes = [leapcube, noleapcube]

    # span=full
    aligned_cubes = mm._align_time_coord(cubes, span='full')
    for cube in aligned_cubes:
        assert cube.coord('time').units.calendar in ("standard", "gregorian")
        assert cube.shape == (367, )
        assert cube[59].coord('time').points == 789  # 29 Feb 1852
    np.ma.is_masked(aligned_cubes[1][366].data)  # outside original range

    result = multi_model_statistics(cubes, span="full", statistics=['mean'])
    result_cube = result['mean']
    assert result_cube[59].data == 2  # looked up nearest neighbour
    assert result_cube[366].data == 1  # outside original range

    # span=overlap
    aligned_cubes = mm._align_time_coord(cubes, span='overlap')
    for cube in aligned_cubes:
        assert cube.coord('time').units.calendar in ("standard", "gregorian")
        assert cube.shape == (365, )
        assert cube[59].coord('time').points == 790  # 1 March 1852

    result = multi_model_statistics(cubes, span="overlap", statistics=['mean'])
    result_cube = result['mean']
    assert result_cube[59].data == 2


def test_remove_fx_variables():
    """Test fx variables are removed from cubes."""
    cube1 = generate_cube_from_dates("monthly")
    fx_cube = generate_cube_from_dates("monthly")
    fx_cube.standard_name = "land_area_fraction"
    add_ancillary_variable(cube1, fx_cube)

    cube2 = generate_cube_from_dates("monthly", fill_val=9)
    result = mm.multi_model_statistics([cube1, cube2],
                                       statistics=['mean'],
                                       span='full')
    assert result['mean'].ancillary_variables() == []


def test_no_warn_model_dim_non_contiguous(recwarn):
    """Test that now warning is raised that model dim is non-contiguous."""
    coord = DimCoord(
        [0.5, 1.5],
        bounds=[[0, 1.], [1., 2.]],
        standard_name='time',
        units='days since 1850-01-01',
    )
    cube1 = iris.cube.Cube([1, 1], dim_coords_and_dims=[(coord, 0)])
    cube2 = iris.cube.Cube([2, 2], dim_coords_and_dims=[(coord, 0)])
    cubes = [cube1, cube2]

    multi_model_statistics(cubes, span="overlap", statistics=['mean'])
    msg = ("Collapsing a non-contiguous coordinate. "
           "Metadata may not be fully descriptive for 'multi-model'.")
    for warning in recwarn:
        assert str(warning.message) != msg


def test_map_to_new_time_int_coords():
    """Test ``_map_to_new_time`` with integer time coords."""
    cube = generate_cube_from_dates('yearly')
    iris.coord_categorisation.add_year(cube, 'time')
    decade_coord = AuxCoord([1850, 1850, 1850], bounds=[[1845, 1855]] * 3,
                            long_name='decade')
    cube.add_aux_coord(decade_coord, 0)
    target_points = [200.0, 500.0, 1000.0]

    out_cube = mm._map_to_new_time(cube, target_points)

    assert_array_allclose(out_cube.data,
                          np.ma.masked_invalid([1.0, 1.0, np.nan]))
    assert_array_allclose(out_cube.coord('time').points, target_points)
    assert_array_allclose(out_cube.coord('year').points,
                          np.ma.masked_invalid([1850, 1851, np.nan]))
    assert_array_allclose(out_cube.coord('decade').points,
                          np.ma.masked_invalid([1850, 1850, np.nan]))
    assert out_cube.coord('year').bounds is None
    assert out_cube.coord('decade').bounds is None
    assert np.issubdtype(out_cube.coord('year').dtype, np.integer)
    assert np.issubdtype(out_cube.coord('decade').dtype, np.integer)


def test_arbitrary_dims_5d(cubes_5d):
    """Test ``multi_model_statistics`` with 5D cubes."""
    stat_cubes = multi_model_statistics(
        cubes_5d,
        span='overlap',
        statistics=['sum'],
    )
    assert len(stat_cubes) == 1
    assert 'sum' in stat_cubes
    stat_cube = stat_cubes['sum']
    assert stat_cube.shape == (1, 1, 1, 1, 1)
    assert_array_allclose(
        stat_cube.data,
        np.ma.array(np.full((1, 1, 1, 1, 1), 3.0)),
    )


def test_arbitrary_dims_2d(cubes_with_arbitrary_dimensions):
    """Test ``multi_model_statistics`` with arbitrary dimensions."""
    stat_cubes = multi_model_statistics(
        cubes_with_arbitrary_dimensions,
        span='overlap',
        statistics=['sum'],
    )
    assert len(stat_cubes) == 1
    assert 'sum' in stat_cubes
    stat_cube = stat_cubes['sum']
    assert stat_cube.shape == (3, 1)
    assert_array_allclose(stat_cube.data, np.ma.array([[0.0], [6.0], [3.0]]))


def test_arbitrary_dims_1d_1(cubes_with_arbitrary_dimensions):
    """Test ``multi_model_statistics`` with arbitrary dimensions."""
    cubes = [cube[0] for cube in cubes_with_arbitrary_dimensions]
    stat_cubes = multi_model_statistics(
        cubes,
        span='overlap',
        statistics=['sum'],
    )
    assert len(stat_cubes) == 1
    assert 'sum' in stat_cubes
    stat_cube = stat_cubes['sum']
    assert stat_cube.shape == (1,)
    assert_array_allclose(stat_cube.data, np.ma.array([0.0]))


def test_arbitrary_dims_1d_3(cubes_with_arbitrary_dimensions):
    """Test ``multi_model_statistics`` with arbitrary dimensions."""
    cubes = [cube[:, 0] for cube in cubes_with_arbitrary_dimensions]
    stat_cubes = multi_model_statistics(
        cubes,
        span='overlap',
        statistics=['sum'],
    )
    assert len(stat_cubes) == 1
    assert 'sum' in stat_cubes
    stat_cube = stat_cubes['sum']
    assert stat_cube.shape == (3,)
    assert_array_allclose(stat_cube.data, np.ma.array([0.0, 6.0, 3.0]))


def test_arbitrary_dims_0d(cubes_with_arbitrary_dimensions):
    """Test ``multi_model_statistics`` with arbitrary dimensions."""
    cubes = [cube[0, 0] for cube in cubes_with_arbitrary_dimensions]
    stat_cubes = multi_model_statistics(
        cubes,
        span='overlap',
        statistics=['sum'],
    )
    assert len(stat_cubes) == 1
    assert 'sum' in stat_cubes
    stat_cube = stat_cubes['sum']
    assert stat_cube.shape == ()
    assert_array_allclose(stat_cube.data, np.ma.array(0.0))


@pytest.mark.parametrize('equal_names', EQUAL_NAMES)
def test_preserve_equal_name_cubes(equal_names):
    """Test ``multi_model_statistics`` with equal-name cubes."""
    all_names = ['var_name', 'standard_name', 'long_name']
    cubes = CubeList(generate_cube_from_dates('monthly') for _ in range(5))

    # Prepare names of input cubes accordingly
    for (idx, cube) in enumerate(cubes):
        for name in all_names:
            if name in equal_names or idx != 0:
                setattr(cube, name, 'air_pressure')
            else:  # Different value for first cube if non-equal name
                setattr(cube, name, None)

    stat_cubes = multi_model_statistics(cubes, span='overlap',
                                        statistics=['sum'])

    assert len(stat_cubes) == 1
    stat_cube = stat_cubes['sum']
    assert_array_allclose(stat_cube.data, np.ma.array([5.0, 5.0, 5.0]))

    for name in all_names:
        assert getattr(stat_cube, name) == 'air_pressure'


@pytest.mark.parametrize('equal_names', EQUAL_NAMES)
def test_equal_name_different_units_cubes(equal_names):
    """Test ``multi_model_statistics`` with equal-name non-equal unit cubes."""
    all_names = ['var_name', 'standard_name', 'long_name']
    cubes = CubeList(generate_cube_from_dates('monthly') for _ in range(5))

    # Prepare names of input cubes accordingly
    cubes[0].units = 'kg'
    for (idx, cube) in enumerate(cubes):
        for name in all_names:
            if name in equal_names or idx != 0:
                setattr(cube, name, 'air_pressure')
            else:  # Different value for first cube if non-equal name
                setattr(cube, name, None)

    msg = (
        "Multi-model statistics failed to merge input cubes into a single "
        "array"
    )
    with pytest.raises(ValueError, match=msg):
        multi_model_statistics(cubes, span='overlap', statistics=['sum'])


def test_equalise_var_metadata():
    """Test ``_equalise_var_metadata``."""
    cubes = CubeList(
        generate_cube_from_dates('monthly', var_name='x') for _ in range(5)
    )

    # Prepare names of input cubes accordingly
    cubes[0].units = 'kg'
    cubes[0].standard_name = 'air_pressure'
    cubes[0].long_name = 'b'
    cubes[1].units = 'kg'
    cubes[1].standard_name = 'air_pressure'
    cubes[1].long_name = 'a'
    cubes[1].var_name = 'y'
    cubes[2].units = 'kg'
    cubes[3].units = 'm'
    cubes[3].long_name = 'X'
    cubes[4].units = 'm'
    cubes[4].long_name = 'X'

    mm._equalise_var_metadata(cubes)

    assert cubes[0].standard_name == 'air_pressure'
    assert cubes[0].long_name == 'a'
    assert cubes[0].var_name == 'x'
    assert cubes[0].units == 'kg'
    assert cubes[1].standard_name == 'air_pressure'
    assert cubes[1].long_name == 'a'
    assert cubes[1].var_name == 'x'
    assert cubes[1].units == 'kg'
    assert cubes[2].standard_name is None
    assert cubes[2].long_name is None
    assert cubes[2].var_name == 'x'
    assert cubes[2].units == 'kg'
    assert cubes[3].standard_name is None
    assert cubes[3].long_name == 'X'
    assert cubes[3].var_name == 'x'
    assert cubes[3].units == 'm'
    assert cubes[4].standard_name is None
    assert cubes[4].long_name == 'X'
    assert cubes[4].var_name == 'x'
    assert cubes[4].units == 'm'


def test_preserve_equal_coordinates():
    """Test ``multi_model_statistics`` with equal input coordinates."""
    cubes = get_cube_for_equal_coords_test(5)
    stat_cubes = multi_model_statistics(cubes, span='overlap',
                                        statistics=['sum'])

    assert len(stat_cubes) == 1
    stat_cube = stat_cubes['sum']
    assert_array_allclose(stat_cube.data, np.ma.array([5.0, 5.0, 5.0]))

    # The equal coordinate 'year' was not changed
    assert stat_cube.coord('year').var_name == 'year'
    assert stat_cube.coord('year').standard_name is None
    assert stat_cube.coord('year').long_name == 'year'
    assert stat_cube.coord('year').attributes == {'test': 1}


def test_preserve_non_equal_coordinates():
    """Test ``multi_model_statistics`` with non_equal input coordinates."""
    cubes = get_cube_for_equal_coords_test(5)

    # Use "circular" attribute for one cube to check that it is set to "False"
    # for each cube
    cubes[2].coord('time').circular = False

    stat_cubes = multi_model_statistics(cubes, span='overlap',
                                        statistics=['sum'])

    assert len(stat_cubes) == 1
    stat_cube = stat_cubes['sum']
    assert_array_allclose(stat_cube.data, np.ma.array([5.0, 5.0, 5.0]))

    # The attributes and circular property of the non-equal coordinate 'time'
    # (due to differing circular) have been removed
    assert stat_cube.coord('time').attributes == {}
    assert stat_cube.coord('time').circular is False

    # The long_name and attributes of the non-equal coordinate 'x' have been
    # removed
    assert stat_cube.coord('x').var_name == 'x'
    assert stat_cube.coord('x').standard_name is None
    assert stat_cube.coord('x').long_name is None
    assert stat_cube.coord('x').attributes == {}


@pytest.mark.parametrize('equal_names', EQUAL_NAMES)
def test_preserve_equal_name_coordinates(equal_names):
    """Test ``multi_model_statistics`` with equal-name coordinates."""
    all_names = ['var_name', 'standard_name', 'long_name']
    cubes = CubeList(generate_cube_from_dates('monthly') for _ in range(5))

    # Prepare names of coordinates of input cubes accordingly
    for (idx, cube) in enumerate(cubes):
        time_coord = cube.coord('time')
        for name in all_names:
            if name in equal_names or idx != 0:
                setattr(time_coord, name, 'time')
            else:  # Different value for first cube if non-equal name
                setattr(time_coord, name, None)

        # Use different coordinate attributes for each cube so the different
        # coordinates are not exactly identical
        time_coord.attributes = {'test': idx}

    stat_cubes = multi_model_statistics(cubes, span='overlap',
                                        statistics=['sum'])

    assert len(stat_cubes) == 1
    stat_cube = stat_cubes['sum']
    assert_array_allclose(stat_cube.data, np.ma.array([5.0, 5.0, 5.0]))

    assert len(stat_cube.coords()) == 1
    time_coord = stat_cube.coords()[0]

    for name in all_names:
        if name in equal_names:
            assert getattr(time_coord, name) == 'time'
        else:
            assert getattr(time_coord, name) is None
    assert time_coord.name() == 'time'
    assert time_coord.units == 'days since 1850-01-01'
    assert time_coord.attributes == {}


def test_ignore_equal_coordinates():
    """Test ``_get_equal_coord_names_metadata``."""
    cubes = CubeList(generate_cube_from_dates('monthly') for _ in range(5))

    equal_coords_metadata = [cubes[0].coord('time').metadata]
    equal_names_metadata = mm._get_equal_coord_names_metadata(
        cubes,
        equal_coords_metadata,
    )

    # The equal_names_metadata dict should be empty since the exactly identical
    # coordinate should be ignored
    assert not equal_names_metadata


@pytest.mark.parametrize('cube_idx', [0, 1, 2, 3, 4])
def test_ignore_duplicate_equal_name_coordinates(cube_idx):
    """Test ``_get_equal_coord_names_metadata``."""
    cubes = CubeList(generate_cube_from_dates('monthly') for _ in range(5))

    # Add duplicate scalar coordinate
    d_coord_0 = AuxCoord(
        0.0,
        var_name='d',
        long_name='d',
        units='m',
        attributes={'test': 1}
    )
    d_coord_1 = AuxCoord(
        1.0,
        var_name='d',
        long_name='d',
        units='m',
    )
    for cube in cubes:
        cube.add_aux_coord(d_coord_0, ())
    cubes[cube_idx].add_aux_coord(d_coord_1, ())

    equal_names_metadata = mm._get_equal_coord_names_metadata(cubes, [])

    # The equal_names_metadata dict should only contain the equal 'time'
    # dimension, not the duplicate dimension
    assert len(equal_names_metadata) == 1
    assert 'time' in equal_names_metadata


def test_ignore_non_existing_coordinates():
    """Test ``_get_equal_coord_names_metadata``."""
    cubes = CubeList(generate_cube_from_dates('monthly') for _ in range(5))

    # Add coordinate only for first cube
    cubes[0].add_aux_coord(AuxCoord(0.0, long_name='x'), ())

    equal_names_metadata = mm._get_equal_coord_names_metadata(cubes, [])

    # The equal_names_metadata dict should only contain the equal 'time'
    # dimension, not the coordinate that only exists for the first cube
    assert len(equal_names_metadata) == 1
    assert 'time' in equal_names_metadata


def test_ignore_coordinates_different_units():
    """Test ``_get_equal_coord_names_metadata``."""
    cubes = CubeList(generate_cube_from_dates('monthly') for _ in range(5))

    # Adapt time units of one cube
    cubes[3].coord('time').units = 'days since 1900-01-01'

    equal_names_metadata = mm._get_equal_coord_names_metadata(cubes, [])

    # The equal_names_metadata dict should be empty since the time units do not
    # match
    assert not equal_names_metadata


def test_empty_input_multi_model_statistics():
    """Check that ``multi_model_statistics`` fails with empty input."""
    msg = "Cannot perform multicube statistics for an empty list of cubes"
    with pytest.raises(ValueError, match=msg):
        mm.multi_model_statistics([], span='full', statistics=['mean'])


def test_empty_input_ensemble_statistics():
    """Check that ``ensemble_statistics`` fails with empty input."""
    msg = "Cannot perform multicube statistics for an empty list of cubes"
    with pytest.raises(ValueError, match=msg):
        mm.ensemble_statistics(
            [], span='full', statistics=['mean'], output_products=[]
        )


STATS = [
    'mean',
    {'operator': 'median'},
    'min',
    'max',
    'p42.314',
    {'operator': 'percentile', 'percent': 42.314},
    'std_dev',
]


@pytest.mark.parametrize('stat', STATS)
@pytest.mark.parametrize(
    'products',
    [
        CubeList([generate_cube_from_dates('monthly')]),
        set([PreprocessorFile(generate_cube_from_dates('monthly'))]),
    ],
)
def test_single_input_multi_model_statistics(products, stat):
    """Check that ``multi_model_statistics`` works with a single cube."""
    output = PreprocessorFile()
    output_products = {'': {mm._get_stat_identifier(stat): output}}
    kwargs = {
        'statistics': [stat],
        'span': 'full',
        'output_products': output_products,
        'keep_input_datasets': False,
    }

    results = mm.multi_model_statistics(products, **kwargs)

    assert len(results) == 1

    if isinstance(results, dict):  # for cube as input
        stat_id = mm._get_stat_identifier(stat)
        cube = results[stat_id]
    else:  # for PreprocessorFile as input
        result = next(iter(results))
        assert len(result.cubes) == 1
        cube = result.cubes[0]

    if stat == 'std_dev':
        assert_array_allclose(
            cube.data, np.ma.masked_invalid([np.nan, np.nan, np.nan])
        )
    else:
        assert_array_allclose(cube.data, np.ma.array([1.0, 1.0, 1.0]))


@pytest.mark.parametrize('stat', STATS)
@pytest.mark.parametrize(
    'products',
    [
        CubeList([generate_cube_from_dates('monthly')]),
        {PreprocessorFile(generate_cube_from_dates('monthly'))},
    ],
)
def test_single_input_ensemble_statistics(products, stat):
    """Check that ``ensemble_statistics`` works with a single cube."""
    stat_id = mm._get_stat_identifier(stat)

    cube = generate_cube_from_dates('monthly')
    attributes = {
        'project': 'project',
        'dataset': 'dataset',
        'exp': 'exp',
        'ensemble': '1',
    }
    products = {PreprocessorFile(cube, attributes=attributes)}
    output = PreprocessorFile()
    output_products = {'project_dataset_exp': {stat_id: output}}
    kwargs = {
        'statistics': [stat],
        'output_products': output_products,
    }

    results = mm.ensemble_statistics(products, **kwargs)

    assert len(results) == 1
    result = next(iter(results))
    assert len(result.cubes) == 1
    cube = result.cubes[0]

    if stat == 'std_dev':
        assert_array_allclose(
            cube.data, np.ma.masked_invalid([np.nan, np.nan, np.nan])
        )
    else:
        assert_array_allclose(cube.data, np.ma.array([1.0, 1.0, 1.0]))


def test_operator_missing_in_stat():
    """Test no operator in stat dict."""
    cubes = CubeList([generate_cube_from_dates('monthly')])
    msg = (
        "`statistic` given as dictionary, but missing required key `operator`"
    )
    with pytest.raises(ValueError) as exc:
        mm.multi_model_statistics(cubes, 'overlap', [{'no': 'operator'}])
    assert msg in str(exc)


@pytest.mark.parametrize(
    'statistic,output',
    [
        ('mean', ('mean', {})),
        ({'operator': 'mean'}, ('mean', {})),
        ({'operator': 'mean', 'weights': False}, ('mean', {'weights': False})),
        ('percentile', ('percentile', {})),
        ({'operator': 'percentile', 'percent': 50},
         ('percentile', {'percent': 50})),
        ({'operator': 'wpercentile', 'weights': False},
         ('wpercentile', {'weights': False})),
        ({'operator': 'wpercentile', 'weights': False, 'percent': 5.0},
         ('wpercentile', {'weights': False, 'percent': 5.0})),
    ]
)
def test_get_operator_and_kwargs(statistic, output):
    """Test ``_get_operator_and_kwargs``."""
    assert mm._get_operator_and_kwargs(statistic) == output


@pytest.mark.parametrize('statistic', [{}, {'no': 'op'}])
def test_get_operator_and_kwargs_operator_missing(statistic):
    """Test ``_get_operator_and_kwargs``."""
    msg = (
        "`statistic` given as dictionary, but missing required key `operator`"
    )
    with pytest.raises(ValueError, match=msg):
        mm._get_operator_and_kwargs(statistic)


@pytest.mark.parametrize(
    'statistic,output',
    [
        ('mean', 'mean'),
        ({'operator': 'mean'}, 'mean'),
        ({'operator': 'mean', 'weights': False}, 'mean'),
        ('percentile', 'percentile'),
        ({'operator': 'percentile', 'percent': 50}, 'percentile50'),
        ({'operator': 'wpercentile', 'weights': False}, 'wpercentile'),
        ({'operator': 'wpercentile', 'weights': False, 'percent': 5.0},
         'wpercentile5.0'),
    ]
)
def test_get_stat_identifier(statistic, output):
    """Test ``_get_stat_identifier``."""
    assert mm._get_stat_identifier(statistic) == output
