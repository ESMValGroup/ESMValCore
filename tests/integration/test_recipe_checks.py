"""Integration tests for :mod:`esmvalcore._recipe_checks`."""
from typing import Any, List
from unittest import mock

import pyesgf.search.results
import pytest

import esmvalcore._recipe_checks as check
import esmvalcore.esgf
from esmvalcore.exceptions import RecipeError
from esmvalcore.preprocessor import PreprocessorFile

ERR_ALL = 'Looked for files matching%s'
ERR_D = ('Looked for files in %s, but did not find any file pattern to match '
         'against')
ERR_F = ('Looked for files matching %s, but did not find any existing input '
         'directory')
ERR_RANGE = 'No input data available for years {} in files:\n{}'
VAR = {
    'filename': 'a/c.nc',
    'frequency': 'mon',
    'short_name': 'tas',
    'start_year': 2020,
    'end_year': 2025,
    'alias': 'alias',
}
FX_VAR = {
    'filename': 'a/b.nc',
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
@mock.patch('esmvalcore._recipe_checks.logger', autospec=True)
def test_data_availability_data(mock_logger, input_files, var, error):
    """Test check for data when data is present."""
    saved_var = dict(var)
    if error is None:
        check.data_availability(input_files, var, None, None)
        mock_logger.error.assert_not_called()
    else:
        with pytest.raises(RecipeError) as rec_err:
            check.data_availability(input_files, var, None, None)
        assert str(rec_err.value) == error
    assert var == saved_var


DATA_AVAILABILITY_NO_DATA: List[Any] = [
    ([], [], None),
    ([], None, None),
    (None, [], None),
    (None, None, None),
    (['dir1'], [], (ERR_D, ['dir1'])),
    (['dir1', 'dir2'], [], (ERR_D, ['dir1', 'dir2'])),
    (['dir1'], None, (ERR_D, ['dir1'])),
    (['dir1', 'dir2'], None, (ERR_D, ['dir1', 'dir2'])),
    ([], ['a*.nc'], (ERR_F, ['a*.nc'])),
    ([], ['a*.nc', 'b*.nc'], (ERR_F, ['a*.nc', 'b*.nc'])),
    (None, ['a*.nc'], (ERR_F, ['a*.nc'])),
    (None, ['a*.nc', 'b*.nc'], (ERR_F, ['a*.nc', 'b*.nc'])),
    (['1'], ['a'], (ERR_ALL, ': 1/a')),
    (['1'], ['a', 'b'], (ERR_ALL, '\n1/a\n1/b')),
    (['1', '2'], ['a'], (ERR_ALL, '\n1/a\n2/a')),
    (['1', '2'], ['a', 'b'], (ERR_ALL, '\n1/a\n1/b\n2/a\n2/b')),
]


@pytest.mark.parametrize('dirnames,filenames,error', DATA_AVAILABILITY_NO_DATA)
@mock.patch('esmvalcore._recipe_checks.logger', autospec=True)
def test_data_availability_no_data(mock_logger, dirnames, filenames, error):
    """Test check for data when no data is present."""
    var = dict(VAR)
    var_no_filename = {
        'frequency': 'mon',
        'short_name': 'tas',
        'start_year': 2020,
        'end_year': 2025,
        'alias': 'alias',
    }
    error_first = ('No input files found for variable %s', var_no_filename)
    error_last = ("Set 'log_level' to 'debug' to get more information", )
    with pytest.raises(RecipeError) as rec_err:
        check.data_availability([], var, dirnames, filenames)
    assert str(rec_err.value) == 'Missing data for alias: tas'
    if error is None:
        assert mock_logger.error.call_count == 2
        errors = [error_first, error_last]
    else:
        assert mock_logger.error.call_count == 3
        errors = [error_first, error, error_last]
    calls = [mock.call(*e) for e in errors]
    assert mock_logger.error.call_args_list == calls
    assert var == VAR


def test_data_availability_nonexistent(tmp_path):
    var = {
        'dataset': 'ABC',
        'short_name': 'tas',
        'frequency': 'mon',
        'start_year': 1990,
        'end_year': 1992,
    }
    result = pyesgf.search.results.FileResult(
        json={
            'dataset_id': 'ABC',
            'project': ['CMIP6'],
            'size': 10,
            'title': 'tas_1990-1992.nc',
        },
        context=None,
    )
    dest_folder = tmp_path
    input_files = [esmvalcore.esgf.ESGFFile([result]).local_file(dest_folder)]
    check.data_availability(input_files, var, dirnames=[], filenames=[])


def test_reference_for_bias_preproc_empty():
    """Test ``reference_for_bias_preproc``."""
    products = {
        PreprocessorFile({'filename': 10}, {}),
        PreprocessorFile({'filename': 20}, {}),
        PreprocessorFile({'filename': 30}, {'trend': {}}),
    }
    check.reference_for_bias_preproc(products)


def test_reference_for_bias_preproc_one_ref():
    """Test ``reference_for_bias_preproc`` with one reference."""
    products = {
        PreprocessorFile({'filename': 90}, {}),
        PreprocessorFile({'filename': 10}, {'bias': {}}),
        PreprocessorFile({'filename': 20}, {'bias': {}}),
        PreprocessorFile({'filename': 30, 'reference_for_bias': True},
                         {'bias': {}})
    }
    check.reference_for_bias_preproc(products)


def test_reference_for_bias_preproc_no_ref():
    """Test ``reference_for_bias_preproc`` with no reference."""
    products = {
        PreprocessorFile({'filename': 90}, {}),
        PreprocessorFile({'filename': 10}, {'bias': {}}),
        PreprocessorFile({'filename': 20}, {'bias': {}}),
        PreprocessorFile({'filename': 30}, {'bias': {}})
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
        PreprocessorFile({'filename': 90}, {}),
        PreprocessorFile({'filename': 10}, {'bias': {}}),
        PreprocessorFile({'filename': 20, 'reference_for_bias': True},
                         {'bias': {}}),
        PreprocessorFile({'filename': 30, 'reference_for_bias': True},
                         {'bias': {}})
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
