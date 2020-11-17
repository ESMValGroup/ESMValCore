"""Functional test for :func:`esmvalcore.preprocessor._multimodel`."""

from pathlib import Path

import iris
import numpy as np
import pytest

from esmvalcore.preprocessor import multi_model_statistics

esmvaltool_sample_data = pytest.importorskip("esmvaltool_sample_data")


def preprocess_data(cubes):
    first_cube = cubes[0]
    t1, t2 = 0, 50  # time-slicing first N items

    # regrid to first cube
    regrid_kwargs = {
        'grid': first_cube,
        'scheme': iris.analysis.Linear(),
    }

    cubes = [cube[t1:t2] for cube in cubes]
    cubes = [cube.regrid(**regrid_kwargs) for cube in cubes]

    return cubes


@pytest.fixture
def timeseries_cubes_amon():
    """Representative timeseries data."""
    timeseries_cubes = esmvaltool_sample_data.load_timeseries_cubes(
        mip_table='Amon')
    timeseries_cubes = preprocess_data(timeseries_cubes)
    return timeseries_cubes


@pytest.fixture
def timeseries_cubes_day():
    """Representative timeseries data."""
    timeseries_cubes = esmvaltool_sample_data.load_timeseries_cubes(
        mip_table='day')
    timeseries_cubes = preprocess_data(timeseries_cubes)
    return timeseries_cubes


def multimodel_test(cubes, span, statistic):
    statistics = [statistic]

    output = multi_model_statistics(cubes, span=span, statistics=statistics)

    return output


def multimodel_regression_test(name, cubes, span):
    statistic = 'mean'
    output = multimodel_test(cubes, span=span, statistic=statistic)

    # this fails for span='overlap', for which `output == cubes` (???)
    assert isinstance(output, dict)

    output_cube = output[statistic]

    # NOTE for the regression test
    # The following test will fail if the data are changed or if the
    # multimodel code changes significantly. To update the data for the
    # regression test, remove the corresponding `.nc` files.
    filename = Path(__file__).with_name(f'{name}-{span}-{statistic}.nc')
    if filename.exists():
        expected_cube = iris.load(str(filename))[0]
        assert np.allclose(output_cube.data, expected_cube.data)
    # else:
    #     iris.save(output_cube, filename)
    #     raise RuntimeError(f'Wrote file {filename.absolute()}')


@pytest.mark.functional
@pytest.mark.parametrize(
    'span',
    (
        pytest.param(
            'overlap',
            marks=pytest.mark.xfail(
                reason=
                '`BUG: multimodel_statistics returns `list` instead of `dict`')
        ),
        # 'overlap',
        'full',
    ))
def test_multimodel_regression_amon(timeseries_cubes_amon, span):
    """Test statistic."""
    cubes = timeseries_cubes_amon
    name = 'timeseries'
    multimodel_regression_test(
        name=name,
        cubes=cubes,
        span=span,
    )


@pytest.mark.functional
@pytest.mark.parametrize(
    'span',
    (
        pytest.param(
            'overlap',
            marks=pytest.mark.xfail(
                reason=
                '`BUG: multimodel_statistics returns `list` instead of `dict`')
        ),
        # 'overlap',
        'full',
    ))
def test_multimodel_regression_day(timeseries_cubes_day, span):
    """Test statistic."""
    cubes = timeseries_cubes_day
    name = 'timeseries'
    multimodel_regression_test(
        name=name,
        cubes=cubes,
        span=span,
    )


@pytest.mark.functional
@pytest.mark.skip('Fix problems with `multimodel_statistics` first')
@pytest.mark.parametrize('span', ('overlap', 'full'))
def test_multimodel_without_vertical_dimension(timeseries_cubes_amon, span):
    """Test statistic without vertical dimension."""
    cubes = timeseries_cubes_amon
    cubes = [cube[0:50, 0] for cube in cubes]
    multimodel_test(cubes, span=span, statistic='mean')


@pytest.mark.functional
@pytest.mark.skip('Fix problems with `multimodel_statistics` first')
@pytest.mark.parametrize('span', ('overlap', 'full'))
def test_multimodel_without_horizontal_dimension(timeseries_cubes_amon, span):
    """Test statistic without horizontal dimension."""
    cubes = timeseries_cubes_amon
    cubes = [cube[0:50, :, 0, 0] for cube in cubes]
    # Coordinate not found error
    # iris.exceptions.CoordinateNotFoundError:
    # 'Expected to find exactly 1 depth coordinate, but found none.'
    multimodel_test(cubes, span=span, statistic='mean')


@pytest.mark.functional
@pytest.mark.skip('Fix problems with `multimodel_statistics` first')
@pytest.mark.parametrize('span', ('overlap', 'full'))
def test_multimodel_only_time_dimension(timeseries_cubes_amon, span):
    """Test statistic without only the time dimension."""
    cubes = timeseries_cubes_amon
    cubes = [cube[0:50, 0, 0, 0] for cube in cubes]
    multimodel_test(cubes, span=span, statistic='mean')


@pytest.mark.functional
@pytest.mark.skip('Fix problems with `multimodel_statistics` first')
@pytest.mark.parametrize('span', ('overlap', 'full'))
def test_multimodel_no_time_dimension(timeseries_cubes_amon, span):
    """Test statistic without time dimension."""
    cubes = timeseries_cubes_amon
    cubes = [cube[0] for cube in cubes]
    # ValueError: Cannot guess bounds for a coordinate of length 1.
    multimodel_test(cubes, span=span, statistic='mean')
