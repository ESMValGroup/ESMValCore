"""Functional test for :func:`esmvalcore.preprocessor._multimodel`."""

from pathlib import Path

import esmvaltool_sample_data
import iris
import numpy as np
import pytest

from esmvalcore.preprocessor import (
    extract_levels,
    multi_model_statistics,
    regrid,
)


def preprocess_data(cubes):
    first_cube = cubes[0]
    t1, t2 = 0, 50  # time-slicing first N items

    # regrid to first cube
    regrid_kwargs = {
        'target_grid': first_cube,
        'scheme': 'linear',
    }
    extract_kwargs = {
        'levels': [100000.0, 92500.0],
        'scheme': 'linear',
    }

    cubes = [cube[t1:t2] for cube in cubes]
    cubes = [regrid(cube, **regrid_kwargs) for cube in cubes]
    cubes = [extract_levels(cube, **extract_kwargs) for cube in cubes]

    return cubes


@pytest.fixture
def timeseries_cubes():
    """Representative timeseries data."""
    timeseries_cubes = esmvaltool_sample_data.load_timeseries_cubes()
    timeseries_cubes = preprocess_data(timeseries_cubes)
    return timeseries_cubes


@pytest.fixture
def map_cubes():
    """Representative map data."""
    map_cubes = esmvaltool_sample_data.load_map_cubes()
    map_cubes = preprocess_data(map_cubes)
    return map_cubes


@pytest.fixture
def profile_cubes():
    """Representative profile data."""
    profile_cubes = esmvaltool_sample_data.load_profile_cubes()
    profile_cubes = preprocess_data(profile_cubes)
    return profile_cubes


def multi_model_test(cubes, span, statistic):
    statistics = [statistic]
    expected_shape = cubes[0].shape

    output = multi_model_statistics(cubes, span=span, statistics=statistics)

    assert isinstance(output, dict)
    assert all(stat in output for stat in statistics)

    output_cube = output[statistic]

    # make sure data are not completely masked
    assert np.all(output_cube.data.mask == False)  # noqa
    assert output_cube.shape == expected_shape

    return output_cube


@pytest.mark.functional
@pytest.mark.parametrize('span', ('overlap', 'full'))
def test_multimodel_overlap(timeseries_cubes, span):
    """Test statistic."""
    cubes = timeseries_cubes
    name = 'timeseries'
    statistic = 'mean'
    output_cube = multi_model_test(cubes, span=span, statistic=statistic)

    # NOTE for the regression test
    # The following test will fail if the data are changed or if the
    # multimodel code changes significantly. To update the data for the
    # regression test, remove the corresponding `.nc` files.
    filename = Path(__file__).with_name(f'{name}-{span}-{statistic}.nc')
    if filename.exists():
        expected_cube = iris.load(str(filename))[0]
        assert np.allclose(output_cube.data, expected_cube.data)
    else:
        iris.save(output_cube, filename)
        raise RuntimeError(f'Wrote file {filename.absolute()}')


@pytest.mark.functional
@pytest.mark.parametrize('span', ('overlap', 'full'))
def test_multimodel_overlap_without_vertical_dimension(timeseries_cubes, span):
    """Test statistic without vertical dimension."""
    cubes = [cube[0:50, 0] for cube in timeseries_cubes]
    statistic = 'mean'
    multi_model_test(cubes, span=span, statistic=statistic)


@pytest.mark.functional
@pytest.mark.parametrize('span', ('overlap', 'full'))
def test_multimodel_overlap_without_horizontal_dimension(
        timeseries_cubes, span):
    """Test statistic without horizontal dimension."""
    cubes = [cube[0:50, :, 0, 0] for cube in timeseries_cubes]
    # Coordinate not found error
    statistic = 'mean'
    with pytest.raises(iris.exceptions.CoordinateNotFoundError):
        # iris.exceptions.CoordinateNotFoundError:
        # 'Expected to find exactly 1 depth coordinate, but found none.'
        multi_model_test(cubes, span=span, statistic=statistic)


@pytest.mark.functional
@pytest.mark.parametrize('span', ('overlap', 'full'))
def test_multimodel_overlap_only_time_dimension(timeseries_cubes, span):
    """Test statistic without only the time dimension."""
    cubes = [cube[0:50, 0, 0, 0] for cube in timeseries_cubes]
    statistic = 'mean'
    multi_model_test(cubes, span=span, statistic=statistic)


@pytest.mark.functional
@pytest.mark.parametrize('span', ('overlap', 'full'))
def test_multimodel_overlap_no_time_dimension(timeseries_cubes, span):
    """Test statistic without time dimension."""
    cubes = [cube[0] for cube in timeseries_cubes]
    statistic = 'mean'
    with pytest.raises(ValueError):
        # ValueError: Cannot guess bounds for a coordinate of length 1.
        multi_model_test(cubes, span=span, statistic=statistic)
