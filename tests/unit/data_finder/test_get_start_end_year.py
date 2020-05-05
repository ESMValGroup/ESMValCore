"""Unit tests for :func:`esmvalcore._data_finder.regrid._stock_cube`"""

import iris
import pytest

from esmvalcore._data_finder import get_start_end_year

FILENAME_CASES = [
    ['var_whatever_1980-1981', 1980, 1981],
    ['var_whatever_1980.nc', 1980, 1980],
    ['var_whatever_19800101-19811231.nc1', 1980, 1981],
    ['var_whatever_19800101.nc', 1980, 1980],
    ['1980-1981_var_whatever.nc', 1980, 1981],
    ['1980_var_whatever.nc', 1980, 1980],
    ['var_control-1980_whatever.nc', 1980, 1980],
    ['19800101-19811231_var_whatever.nc', 1980, 1981],
    ['19800101_var_whatever.nc', 1980, 1980],
    ['var_control-19800101_whatever.nc', 1980, 1980],
    ['19800101_var_control-1950_whatever.nc', 1980, 1980],
    ['var_control-1950_whatever_19800101.nc', 1980, 1980],
]


@pytest.mark.parametrize('case', FILENAME_CASES)
def test_get_start_end_year(case):
    """Tests for get_start_end_year function"""
    filename, case_start, case_end = case
    start, end = get_start_end_year(filename)
    assert case_start == start
    assert case_end == end


def test_read_time_from_cube(monkeypatch, tmp_path):
    """Try to get time from cube if no date in filename"""
    monkeypatch.chdir(tmp_path)
    temp_file = 'test.nc'
    cube = iris.cube.Cube([0, 0], var_name='var')
    time = iris.coords.DimCoord([0, 366],
                                'time',
                                units='days since 1990-01-01')
    cube.add_dim_coord(time, 0)
    iris.save(cube, temp_file)
    start, end = get_start_end_year(temp_file)
    assert start == 1990
    assert end == 1991


def test_fails_if_no_date_present():
    """Test raises if no date is present"""
    with pytest.raises((ValueError, OSError)):
        get_start_end_year('var_whatever')
