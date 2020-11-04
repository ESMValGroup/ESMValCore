"""System test for :func:`esmvalcore.preprocessor._multimodel`."""

import esmvaltool_sample_data
import pytest

from esmvalcore.preprocessor import (
    extract_levels,
    multi_model_statistics,
    regrid,
)


def preprocess_data(data):
    regrid_kwargs = {
        'target_grid': '2.5x2.5',
        'scheme': 'linear',
    }
    extract_kwargs = {
        'levels': [100000.0, 92500.0],
        'scheme': 'linear',
    }

    data = [regrid(d, **regrid_kwargs) for d in data]
    data = [extract_levels(d, **extract_kwargs) for d in data]

    return data


def get_timeseries_data():
    """Representative timeseries data."""
    timeseries_data = esmvaltool_sample_data.load_timeseries_data()
    timeseries_data = preprocess_data(timeseries_data)
    return timeseries_data


def get_map_data():
    """Representative timeseries data."""
    map_data = esmvaltool_sample_data.load_map_data()
    map_data = preprocess_data(map_data)
    return map_data


def get_profile_data():
    """Representative timeseries data."""
    profile_data = esmvaltool_sample_data.load_profile_data()
    profile_data = preprocess_data(profile_data)
    return profile_data


data_list = [
    get_timeseries_data(),
    # get_map_data(),
    # get_profile_data(),
]


@pytest.mark.parametrize('data', data_list)
def test_multimodel_overlap_mean(data):
    """Test statistic."""
    stat = multi_model_statistics(data, span='overlap', statistics=['mean'])


@pytest.mark.skip('Takes too long to run')
@pytest.mark.parametrize('data', data_list)
def test_multimodel_full_mean(data):
    """Test statistic."""
    stat = multi_model_statistics(data, span='full', statistics=['mean'])


# mean, median, min, max, std, p75
