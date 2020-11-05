"""Functional test for :func:`esmvalcore.preprocessor._multimodel`."""

import esmvaltool_sample_data
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


@pytest.mark.parametrize('span', ('overlap', 'full'))
def test_multimodel_overlap(timeseries_cubes, span):
    """Test statistic."""
    cubes = timeseries_cubes
    statistic = 'mean'
    statistics = [statistic]
    expected_shape = cubes[0].shape

    output = multi_model_statistics(cubes, span=span, statistics=statistics)

    assert isinstance(output, dict)
    assert all(stat in output for stat in statistics)

    output_cube = output[statistic]

    # make sure data are not completely masked
    assert np.all(output_cube.data.mask == False)  # noqa

    assert output_cube.shape == expected_shape
