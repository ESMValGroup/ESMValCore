"""Unit tests for time related functions in `esmvalcore.local`."""
from pathlib import Path

import iris
import pyesgf
import pytest

from esmvalcore.esgf import ESGFFile
from esmvalcore.local import (
    LocalFile,
    _dates_to_timerange,
    _get_start_end_date,
    _get_start_end_year,
    _replace_years_with_timerange,
    _truncate_dates,
)


def _get_esgf_file(path):
    """Get ESGFFile object."""
    result = pyesgf.search.results.FileResult(
        json={
            'dataset_id': 'CMIP6.ABC.v1|something.org',
            'dataset_id_template_': ["%(mip_era)s.%(source_id)s"],
            'project': ['CMIP6'],
            'size': 10,
            'title': path,
        },
        context=None,
    )
    return ESGFFile([result])


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
    ['tas_A1.20C3M_1.CCSM.atmm.1990-01_cat_1999-12.nc', 1990, 1999],
    ['E5sf00_1M_1940_032.grb', 1940, 1940],
    ['E5sf00_1D_1998-04_167.grb', 1998, 1998],
    ['E5sf00_1H_1986-04-11_167.grb', 1986, 1986],
    ['E5sf00_1M_1940-1941_032.grb', 1940, 1941],
    ['E5sf00_1D_1998-01_1999-12_167.grb', 1998, 1999],
    ['E5sf00_1H_2000-01-01_2001-12-31_167.grb', 2000, 2001],
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
    ['tas_A1.20C3M_1.CCSM.atmm.1990-01_cat_1999-12.nc', '199001', '199912'],
    ['E5sf00_1M_1940_032.grb', '1940', '1940'],
    ['E5sf00_1D_1998-04_167.grb', '199804', '199804'],
    ['E5sf00_1H_1986-04-11_167.grb', '19860411', '19860411'],
    ['E5sf00_1M_1940-1941_032.grb', '1940', '1941'],
    ['E5sf00_1D_1998-01_1999-12_167.grb', '199801', '199912'],
    ['E5sf00_1H_2000-01-01_2001-12-31_167.grb', '20000101', '20011231'],
]


@pytest.mark.parametrize('case', FILENAME_CASES)
def test_get_start_end_year(case):
    """Tests for _get_start_end_year function."""
    filename, case_start, case_end = case

    # If the filename is inconclusive or too difficult we resort to reading the
    # file, which fails here because the file is not there.
    if case_start is None and case_end is None:
        with pytest.raises(ValueError):
            _get_start_end_year(filename)
        with pytest.raises(ValueError):
            _get_start_end_year(Path(filename))
        with pytest.raises(ValueError):
            _get_start_end_year(LocalFile(filename))
        with pytest.raises(ValueError):
            _get_start_end_year(_get_esgf_file(filename))

    else:
        start, end = _get_start_end_year(filename)
        assert case_start == start
        assert case_end == end
        start, end = _get_start_end_year(Path(filename))
        assert case_start == start
        assert case_end == end
        start, end = _get_start_end_year(LocalFile(filename))
        assert case_start == start
        assert case_end == end
        start, end = _get_start_end_year(_get_esgf_file(filename))
        assert case_start == start
        assert case_end == end


@pytest.mark.parametrize('case', FILENAME_DATE_CASES)
def test_get_start_end_date(case):
    """Tests for _get_start_end_date function."""
    filename, case_start, case_end = case

    # If the filename is inconclusive or too difficult we resort to reading the
    # file, which fails here because the file is not there.
    if case_start is None and case_end is None:
        with pytest.raises(ValueError):
            _get_start_end_date(filename)
        with pytest.raises(ValueError):
            _get_start_end_date(Path(filename))
        with pytest.raises(ValueError):
            _get_start_end_date(LocalFile(filename))
        with pytest.raises(ValueError):
            _get_start_end_date(_get_esgf_file(filename))

    else:
        start, end = _get_start_end_date(filename)
        assert case_start == start
        assert case_end == end
        start, end = _get_start_end_date(Path(filename))
        assert case_start == start
        assert case_end == end
        start, end = _get_start_end_date(LocalFile(filename))
        assert case_start == start
        assert case_end == end
        start, end = _get_start_end_date(_get_esgf_file(filename))
        assert case_start == start
        assert case_end == end


def test_read_years_from_cube(monkeypatch, tmp_path):
    """Try to get years from cube if no date in filename."""
    monkeypatch.chdir(tmp_path)
    temp_file = LocalFile('test.nc')
    cube = iris.cube.Cube([0, 0], var_name='var')
    time = iris.coords.DimCoord([0, 366],
                                'time',
                                units='days since 1990-01-01')
    cube.add_dim_coord(time, 0)
    iris.save(cube, temp_file)
    start, end = _get_start_end_year(temp_file)
    assert start == 1990
    assert end == 1991


def test_read_datetime_from_cube(monkeypatch, tmp_path):
    """Try to get datetime from cube if no date in filename."""
    monkeypatch.chdir(tmp_path)
    temp_file = 'test.nc'
    cube = iris.cube.Cube([0, 0], var_name='var')
    time = iris.coords.DimCoord([0, 366],
                                'time',
                                units='days since 1990-01-01')
    cube.add_dim_coord(time, 0)
    iris.save(cube, temp_file)
    start, end = _get_start_end_date(temp_file)
    assert start == '19900101'
    assert end == '19910102'


def test_raises_if_unable_to_deduce(monkeypatch, tmp_path):
    """Try to get time from cube if no date in filename."""
    monkeypatch.chdir(tmp_path)
    temp_file = 'test.nc'
    cube = iris.cube.Cube([0, 0], var_name='var')
    iris.save(cube, temp_file)
    with pytest.raises(ValueError):
        _get_start_end_date(temp_file)
    with pytest.raises(ValueError):
        _get_start_end_year(temp_file)


def test_fails_if_no_date_present():
    """Test raises if no date is present."""
    with pytest.raises((ValueError)):
        _get_start_end_date('var_whatever')
    with pytest.raises((ValueError)):
        _get_start_end_year('var_whatever')


def test_get_timerange_from_years():
    """Test a `timerange` tag with value `start_year/end_year` can be built
    from tags `start_year` and `end_year`."""
    variable = {'start_year': 2000, 'end_year': 2002}

    _replace_years_with_timerange(variable)

    assert 'start_year' not in variable
    assert 'end_year' not in variable
    assert variable['timerange'] == '2000/2002'


def test_get_timerange_from_start_year():
    """Test a `timerange` tag with value `start_year/start_year` can be built
    from tag `start_year` when an `end_year` is not given."""
    variable = {'start_year': 2000}

    _replace_years_with_timerange(variable)

    assert 'start_year' not in variable
    assert variable['timerange'] == '2000/2000'


def test_get_timerange_from_end_year():
    """Test a `timerange` tag with value `end_year/end_year` can be built from
    tag `end_year` when a `start_year` is not given."""
    variable = {'end_year': 2002}

    _replace_years_with_timerange(variable)

    assert 'end_year' not in variable
    assert variable['timerange'] == '2002/2002'


TEST_DATES_TO_TIMERANGE = [
    (2000, 2000, '2000/2000'),
    (1, 2000, '0001/2000'),
    (2000, 1, '2000/0001'),
    (1, 2, '0001/0002'),
    ('2000', '2000', '2000/2000'),
    ('1', '2000', '0001/2000'),
    (2000, '1', '2000/0001'),
    ('1', 2, '0001/0002'),
    ('*', '*', '*/*'),
    (2000, '*', '2000/*'),
    ('2000', '*', '2000/*'),
    (1, '*', '0001/*'),
    ('1', '*', '0001/*'),
    ('*', 2000, '*/2000'),
    ('*', '2000', '*/2000'),
    ('*', 1, '*/0001'),
    ('*', '1', '*/0001'),
    ('P5Y', 'P5Y', 'P5Y/P5Y'),
    (2000, 'P5Y', '2000/P5Y'),
    ('2000', 'P5Y', '2000/P5Y'),
    (1, 'P5Y', '0001/P5Y'),
    ('1', 'P5Y', '0001/P5Y'),
    ('P5Y', 2000, 'P5Y/2000'),
    ('P5Y', '2000', 'P5Y/2000'),
    ('P5Y', 1, 'P5Y/0001'),
    ('P5Y', '1', 'P5Y/0001'),
    ('*', 'P5Y', '*/P5Y'),
    ('P5Y', '*', 'P5Y/*'),
]


@pytest.mark.parametrize('start_date,end_date,expected_timerange',
                         TEST_DATES_TO_TIMERANGE)
def test_dates_to_timerange(start_date, end_date, expected_timerange):
    """Test ``_dates_to_timerange``."""
    timerange = _dates_to_timerange(start_date, end_date)
    assert timerange == expected_timerange


TEST_TRUNCATE_DATES = [
    ('2000', '2000', (2000, 2000)),
    ('200001', '2000', (2000, 2000)),
    ('2000', '200001', (2000, 2000)),
    ('200001', '2000', (2000, 2000)),
    ('200001', '200001', (200001, 200001)),
    ('20000102', '200001', (200001, 200001)),
    ('200001', '20000102', (200001, 200001)),
    ('20000102', '20000102', (20000102, 20000102)),
    ('20000102T23:59:59', '20000102', (20000102, 20000102)),
    ('20000102', '20000102T23:59:59', (20000102, 20000102)),
    ('20000102T235959', '20000102T01:02:03', (20000102235959, 20000102010203)),
]


@pytest.mark.parametrize('date,date_file,expected_output', TEST_TRUNCATE_DATES)
def test_truncate_dates(date, date_file, expected_output):
    """Test ``_truncate_dates``."""
    output = _truncate_dates(date, date_file)
    assert output == expected_output
