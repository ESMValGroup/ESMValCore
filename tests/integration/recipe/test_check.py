"""Integration tests for :mod:`esmvalcore._recipe.check`."""
import os.path
from pathlib import Path
from typing import Any, List
from unittest import mock

import pyesgf.search.results
import pytest

import esmvalcore._recipe.check
import esmvalcore.esgf
from esmvalcore._recipe import check
from esmvalcore.dataset import Dataset
from esmvalcore.exceptions import RecipeError
from esmvalcore.preprocessor import PreprocessorFile

ERR_ALL = 'Looked for files matching%s'
ERR_RANGE = 'No input data available for years {} in files:\n{}'
VAR = {
    'frequency': 'mon',
    'short_name': 'tas',
    'timerange': '2020/2025',
    'alias': 'alias',
    'start_year': 2020,
    'end_year': 2025
}
FX_VAR = {
    'frequency': 'fx',
    'short_name': 'areacella',
}
FILES = [
    'a/b/c_20200101-20201231',
    'a/b/c_20210101-20211231',
    'a/b/c_20220101-20221231',
    'a/b/c_20230101-20231231',
    'a/b/c_20240101-20241231',
    'a/b/c_20250101-20251231',
]

DATA_AVAILABILITY_DATA = [
    (FILES, dict(VAR), None),
    (FILES, dict(FX_VAR), None),
    (FILES[1:], dict(VAR), ERR_RANGE.format('2020', "\n".join(FILES[1:]))),
    (FILES[:-1], dict(VAR), ERR_RANGE.format('2025', "\n".join(FILES[:-1]))),
    (FILES[:-3], dict(VAR), ERR_RANGE.format('2023-2025',
                                             "\n".join(FILES[:-3]))),
    ([FILES[1]] + [FILES[3]], dict(VAR),
     ERR_RANGE.format('2020, 2022, 2024-2025',
                      "\n".join([FILES[1]] + [FILES[3]]))),
]


@pytest.mark.parametrize('input_files,var,error', DATA_AVAILABILITY_DATA)
@mock.patch('esmvalcore._recipe.check.logger', autospec=True)
def test_data_availability_data(mock_logger, input_files, var, error):
    """Test check for data when data is present."""
    dataset = Dataset(**var)
    dataset.files = [Path(f) for f in input_files]
    if error is None:
        check.data_availability(dataset)
        mock_logger.error.assert_not_called()
    else:
        with pytest.raises(RecipeError) as rec_err:
            check.data_availability(dataset)
        assert str(rec_err.value) == error
    assert dataset.facets == var


DATA_AVAILABILITY_NO_DATA: List[Any] = [
    ([], [], None),
    ([''], ['a*.nc'], (ERR_ALL, ': a*.nc')),
    ([''], ['a*.nc', 'b*.nc'], (ERR_ALL, '\na*.nc\nb*.nc')),
    (['1'], ['a'], (ERR_ALL, ': 1/a')),
    (['1'], ['a', 'b'], (ERR_ALL, '\n1/a\n1/b')),
    (['1', '2'], ['a'], (ERR_ALL, '\n1/a\n2/a')),
    (['1', '2'], ['a', 'b'], (ERR_ALL, '\n1/a\n1/b\n2/a\n2/b')),
]


@pytest.mark.parametrize('dirnames,filenames,error', DATA_AVAILABILITY_NO_DATA)
@mock.patch('esmvalcore._recipe.check.logger', autospec=True)
def test_data_availability_no_data(mock_logger, dirnames, filenames, error):
    """Test check for data when no data is present."""
    facets = {
        'frequency': 'mon',
        'short_name': 'tas',
        'timerange': '2020/2025',
        'alias': 'alias',
        'start_year': 2020,
        'end_year': 2025
    }
    dataset = Dataset(**facets)
    dataset.files = []
    dataset._file_globs = [
        os.path.join(d, f) for d in dirnames for f in filenames
    ]
    error_first = ('No input files found for %s', dataset)
    error_last = ("Set 'log_level' to 'debug' to get more information", )
    with pytest.raises(RecipeError) as rec_err:
        check.data_availability(dataset)
    assert str(rec_err.value) == 'Missing data for Dataset: tas'
    if error is None:
        assert mock_logger.error.call_count == 2
        errors = [error_first, error_last]
    else:
        assert mock_logger.error.call_count == 3
        errors = [error_first, error, error_last]
    calls = [mock.call(*e) for e in errors]
    assert mock_logger.error.call_args_list == calls
    assert dataset.facets == facets


GOOD_TIMERANGES = [
    '*',
    '1990/1992',
    '19900101/19920101',
    '19900101T12H00M00S/19920101T12H00M00',
    '1990/*',
    '*/1992',
    '1990/P2Y',
    '19900101/P2Y2M1D',
    '19900101TH00M00S/P2Y2M1DT12H00M00S',
    'P2Y/1992',
    'P2Y2M1D/19920101',
    'P2Y2M1D/19920101T12H00M00S',
    'P2Y/*',
    'P2Y2M1D/*',
    'P2Y21DT12H00M00S/*',
    '*/P2Y',
    '*/P2Y2M1D',
    '*/P2Y21DT12H00M00S',
    '1/301',
    '1/*',
    '*/301',
]


@pytest.mark.parametrize('timerange', GOOD_TIMERANGES)
def test_valid_time_selection(timerange):
    """Check that good definitions do not raise anything."""
    check.valid_time_selection(timerange)


BAD_TIMERANGES = [
    ('randomnonsense',
     'Invalid value encountered for `timerange`. Valid values must be '
     "separated by `/`. Got ['randomnonsense'] instead."),
    ('199035345/19923463164526',
     'Invalid value encountered for `timerange`. Valid value must follow '
     "ISO 8601 standard for dates and duration periods, or be set to '*' "
     "to load available years. Got ['199035345', '19923463164526'] instead."),
    ('P11Y/P42Y', 'Invalid value encountered for `timerange`. Cannot set both '
     'the beginning and the end as duration periods.'),
]


@pytest.mark.parametrize('timerange,message', BAD_TIMERANGES)
def test_valid_time_selection_rejections(timerange, message):
    """Check that bad definitions raise RecipeError."""
    with pytest.raises(check.RecipeError) as rec_err:
        check.valid_time_selection(timerange)
    assert str(rec_err.value) == message


def test_differing_timeranges(caplog):
    timeranges = set()
    timeranges.add('1950/1951')
    timeranges.add('1950/1952')
    required_variables = [
        {
            'short_name': 'rsdscs',
            'timerange': '1950/1951'
        },
        {
            'short_name': 'rsuscs',
            'timerange': '1950/1952'
        },
    ]
    with pytest.raises(ValueError) as exc:
        check.differing_timeranges(
            timeranges, required_variables)
    expected_log = (
        f"Differing timeranges with values {timeranges} "
        "found for required variables "
        "[{'short_name': 'rsdscs', 'timerange': '1950/1951'}, "
        "{'short_name': 'rsuscs', 'timerange': '1950/1952'}]. "
        "Set `timerange` to a common value."
    )

    assert expected_log in str(exc.value)


def test_data_availability_nonexistent(tmp_path):
    var = {
        'dataset': 'ABC',
        'short_name': 'tas',
        'frequency': 'mon',
        'timerange': '1990/1992',
        'start_year': 1990,
        'end_year': 1992
    }
    result = pyesgf.search.results.FileResult(
        json={
            'dataset_id': 'CMIP6.ABC.v1|something.org',
            'dataset_id_template_': ["%(mip_era)s.%(source_id)s"],
            'project': ['CMIP6'],
            'size': 10,
            'title': 'tas_1990-1992.nc',
        },
        context=None,
    )
    dest_folder = tmp_path
    input_files = [esmvalcore.esgf.ESGFFile([result]).local_file(dest_folder)]
    dataset = Dataset(**var)
    dataset.files = input_files
    check.data_availability(dataset)


def test_reference_for_bias_preproc_empty():
    """Test ``reference_for_bias_preproc``."""
    products = {
        PreprocessorFile(filename=10),
        PreprocessorFile(filename=20),
        PreprocessorFile(filename=30),
    }
    check.reference_for_bias_preproc(products)


def test_reference_for_bias_preproc_one_ref():
    """Test ``reference_for_bias_preproc`` with one reference."""
    products = {
        PreprocessorFile(filename=90),
        PreprocessorFile(filename=10,
                         settings={'bias': {}}),
        PreprocessorFile(filename=20,
                         settings={'bias': {}}),
        PreprocessorFile(filename=30,
                         settings={'bias': {}},
                         attributes={'reference_for_bias': True}),
    }
    check.reference_for_bias_preproc(products)


def test_reference_for_bias_preproc_no_ref():
    """Test ``reference_for_bias_preproc`` with no reference."""
    products = {
        PreprocessorFile(filename=90),
        PreprocessorFile(filename=10,
                         settings={'bias': {}}),
        PreprocessorFile(filename=20,
                         settings={'bias': {}}),
        PreprocessorFile(filename=30,
                         settings={'bias': {}})
    }
    with pytest.raises(RecipeError) as rec_err:
        check.reference_for_bias_preproc(products)

    # Note: checking the message directly does not work due to the unknown
    # (machine-dependent) ordering of products in the set
    assert ("Expected exactly 1 dataset with 'reference_for_bias: true' in "
            "products\n[") in str(rec_err.value)
    assert '10' in str(rec_err.value)
    assert '20' in str(rec_err.value)
    assert '30' in str(rec_err.value)
    assert '90' not in str(rec_err.value)
    assert ("],\nfound 0. Please also ensure that the reference dataset is "
            "not excluded with the 'exclude' option") in str(rec_err.value)


def test_reference_for_bias_preproc_two_refs():
    """Test ``reference_for_bias_preproc`` with two references."""
    products = {
        PreprocessorFile(filename=90),
        PreprocessorFile(filename=10, settings={'bias': {}}),
        PreprocessorFile(filename=20,
                         attributes={'reference_for_bias': True},
                         settings={'bias': {}}),
        PreprocessorFile(filename=30,
                         attributes={'reference_for_bias': True},
                         settings={'bias': {}})
    }
    with pytest.raises(RecipeError) as rec_err:
        check.reference_for_bias_preproc(products)

    # Note: checking the message directly does not work due to the unknown
    # (machine-dependent) ordering of products in the set
    assert ("Expected exactly 1 dataset with 'reference_for_bias: true' in "
            "products\n[") in str(rec_err.value)
    assert '10' in str(rec_err.value)
    assert '20' in str(rec_err.value)
    assert '30' in str(rec_err.value)
    assert '90' not in str(rec_err.value)
    assert "],\nfound 2:\n[" in str(rec_err.value)
    assert ("].\nPlease also ensure that the reference dataset is "
            "not excluded with the 'exclude' option") in str(rec_err.value)


INVALID_MM_SETTINGS = {
        'wrong_parametre': 'wrong',
        'statistics': ['wrong'],
        'span': 'wrong',
        'groupby': 'wrong',
        'keep_input_datasets': 'wrong',
        'ignore_scalar_coords': 'wrong',
    }


def test_invalid_multi_model_span():
    with pytest.raises(RecipeError) as rec_err:
        check._verify_span_value(INVALID_MM_SETTINGS['span'])
    assert str(rec_err.value) == (
        "Invalid value encountered for `span` in preprocessor "
        "`multi_model_statistics`. Valid values are ('overlap', 'full')."
        "Got wrong."
    )


def test_invalid_multi_model_groupy():
    with pytest.raises(RecipeError) as rec_err:
        check._verify_groupby(INVALID_MM_SETTINGS['groupby'])
    assert str(rec_err.value) == (
        'Invalid value encountered for `groupby` in preprocessor '
        '`multi_model_statistics`.`groupby` must be defined '
        'as a list. Got wrong.'
    )


def test_invalid_multi_model_keep_input():
    with pytest.raises(RecipeError) as rec_err:
        check._verify_keep_input_datasets(
            INVALID_MM_SETTINGS['keep_input_datasets'])
    assert str(rec_err.value) == (
        'Invalid value encountered for `keep_input_datasets`.'
        'Must be defined as a boolean (true or false). Got wrong.')


def test_invalid_multi_model_ignore_scalar_coords():
    with pytest.raises(RecipeError) as rec_err:
        check._verify_ignore_scalar_coords(
            INVALID_MM_SETTINGS['ignore_scalar_coords'])
    assert str(rec_err.value) == (
        'Invalid value encountered for `ignore_scalar_coords`.'
        'Must be defined as a boolean (true or false). Got wrong.')
