"""Unit tests for :func:`esmvalcore._data_finder.regrid._stock_cube`"""

import iris
import pytest

from esmvalcore._data_finder import (
    _get_timerange_from_years,
    get_start_end_date,
    get_start_end_year,
)

FILENAME_CASES = [
    ['var_whatever_1980-1981', 1980, 1981],
    ['var_whatever_1980.nc', 1980, 1980],
    ['a.b.x_yz_185001-200512.nc', 1850, 2005],
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
    ['CM61-LR-hist-03.1950_18500101_19491231_1M_concbc.nc', 1850, 1949],
    [
        'icon-2.6.1_atm_amip_R2B5_r1v1i1p1l1f1_phy_3d_ml_20150101T000000Z.nc',
        2015, 2015
    ],
    ['pr_A1.186101-200012.nc', 1861, 2000],
    ['tas_A1.20C3M_1.CCSM.atmm.1990-01_cat_1999-12.nc', None, None],
]

FILENAME_DATE_CASES = [
    ['var_whatever_1980-1981', '1980', '1981'],
    ['var_whatever_1980.nc', '1980', '1980'],
    ['a.b.x_yz_185001-200512.nc', '185001', '200512'],
    ['var_whatever_19800101-19811231.nc1', '19800101', '19811231'],
    ['var_whatever_19800101.nc', '19800101', '19800101'],
    ['1980-1981_var_whatever.nc', '1980', '1981'],
    ['1980_var_whatever.nc', '1980', '1980'],
    ['var_control-1980_whatever.nc', '1980', '1980'],
    ['19800101-19811231_var_whatever.nc', '19800101', '19811231'],
    ['19800101_var_whatever.nc', '19800101', '19800101'],
    ['var_control-19800101_whatever.nc', '19800101', '19800101'],
    ['19800101_var_control-1950_whatever.nc', '19800101', '19800101'],
    ['var_control-1950_whatever_19800101.nc', '19800101', '19800101'],
    [
        'CM61-LR-hist-03.1950_18500101_19491231_1M_concbc.nc', '18500101',
        '19491231'
    ],
    [
        'icon-2.6.1_atm_amip_R2B5_r1v1i1p1l1f1_phy_3d_ml_20150101T000000Z.nc',
        '20150101T000000Z', '20150101T000000Z'
    ],
    ['pr_A1.186101-200012.nc', '186101', '200012'],
    ['tas_A1.20C3M_1.CCSM.atmm.1990-01_cat_1999-12.nc', None, None],
]


@pytest.mark.parametrize('case', FILENAME_CASES)
def test_get_start_end_year(case):
    """Tests for get_start_end_year function."""
    filename, case_start, case_end = case
    if case_start is None and case_end is None:
        # If the filename is inconclusive or too difficult
        # we resort to reading the file, which fails here
        # because the file is not there.
        with pytest.raises(ValueError):
            get_start_end_year(filename)
    else:
        start, end = get_start_end_year(filename)
        assert case_start == start
        assert case_end == end


@pytest.mark.parametrize('case', FILENAME_DATE_CASES)
def test_get_start_end_date(case):
    """Tests for get_start_end_date function."""
    filename, case_start, case_end = case
    if case_start is None and case_end is None:
        # If the filename is inconclusive or too difficult
        # we resort to reading the file, which fails here
        # because the file is not there.
        with pytest.raises(ValueError):
            get_start_end_date(filename)
    else:
        start, end = get_start_end_date(filename)
        assert case_start == start
        assert case_end == end


def test_read_time_from_cube(monkeypatch, tmp_path):
    """Try to get time from cube if no date in filename."""
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


def test_read_datetime_from_cube(monkeypatch, tmp_path):
    """Try to get time from cube if no date in filename."""
    monkeypatch.chdir(tmp_path)
    temp_file = 'test.nc'
    cube = iris.cube.Cube([0, 0], var_name='var')
    time = iris.coords.DimCoord([0, 366],
                                'time',
                                units='days since 1990-01-01')
    cube.add_dim_coord(time, 0)
    iris.save(cube, temp_file)
    start, end = get_start_end_date(temp_file)
    assert start == '19900101'
    assert end == '19910102'


def test_raises_if_unable_to_deduce(monkeypatch, tmp_path):
    """Try to get time from cube if no date in filename."""
    monkeypatch.chdir(tmp_path)
    temp_file = 'test.nc'
    cube = iris.cube.Cube([0, 0], var_name='var')
    iris.save(cube, temp_file)
    with pytest.raises(ValueError):
        get_start_end_date(temp_file)


def test_fails_if_no_date_present():
    """Test raises if no date is present."""
    with pytest.raises((ValueError, OSError)):
        get_start_end_date('var_whatever')
        get_start_end_year('var_whatever')


def test_get_timerange_from_years():
    """Test a `timerange` tag with value `start_year/end_year` can be built
    from tags `start_year` and `end_year`."""
    variable = {
        'start_year': 2000,
        'end_year': 2002}

    _get_timerange_from_years(variable)

    assert 'start_year' not in variable
    assert 'end_year' not in variable
    assert variable['timerange'] == '2000/2002'


def test_get_timerange_from_start_year():
    """Test a `timerange` tag with value `start_year/start_year` can be built
    from tag `start_year` when an `end_year` is not given."""
    variable = {
        'start_year': 2000
    }

    _get_timerange_from_years(variable)

    assert 'start_year' not in variable
    assert variable['timerange'] == '2000/2000'


def test_get_timerange_from_end_year():
    """Test a `timerange` tag with value `end_year/end_year` can be built from
    tag `end_year` when a `start_year` is not given."""
    variable = {
        'end_year': 2002
    }

    _get_timerange_from_years(variable)

    assert 'end_year' not in variable
    assert variable['timerange'] == '2002/2002'
