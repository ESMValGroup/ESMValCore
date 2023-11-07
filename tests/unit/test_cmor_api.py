# flake8: noqa
from unittest.mock import sentinel

import esmvalcore.cmor.check
import esmvalcore.cmor.fix
import esmvalcore.cmor.fixes
import esmvalcore.cmor.table
from esmvalcore.cmor.check import (
    CheckLevels,
    CMORCheck,
    CMORCheckError,
    cmor_check,
    cmor_check_data,
    cmor_check_metadata,
)


def test_cmor_check_metadata(mocker):
    """Test ``cmor_check_metadata``"""
    mock_get_cmor_checker = mocker.patch.object(
        esmvalcore.cmor.check, '_get_cmor_checker', autospec=True
    )
    (
        mock_get_cmor_checker.return_value.return_value.check_metadata.
        return_value
    ) = sentinel.checked_cube

    cube = cmor_check_metadata(
        sentinel.cube,
        sentinel.cmor_table,
        sentinel.mip,
        sentinel.short_name,
        sentinel.frequency,
        check_level=sentinel.check_level,
    )

    mock_get_cmor_checker.assert_called_once_with(
        sentinel.cmor_table,
        sentinel.mip,
        sentinel.short_name,
        sentinel.frequency,
        check_level=sentinel.check_level,
    )
    mock_get_cmor_checker.return_value.assert_called_once_with(sentinel.cube)
    (
        mock_get_cmor_checker.return_value.return_value.check_metadata.
        assert_called_once_with()
    )
    assert cube == sentinel.checked_cube


def test_cmor_check_data(mocker):
    """Test ``cmor_check_data``"""
    mock_get_cmor_checker = mocker.patch.object(
        esmvalcore.cmor.check, '_get_cmor_checker', autospec=True
    )
    (
        mock_get_cmor_checker.return_value.return_value.check_data.
        return_value
    ) = sentinel.checked_cube

    cube = cmor_check_data(
        sentinel.cube,
        sentinel.cmor_table,
        sentinel.mip,
        sentinel.short_name,
        sentinel.frequency,
        check_level=sentinel.check_level,
    )

    mock_get_cmor_checker.assert_called_once_with(
        sentinel.cmor_table,
        sentinel.mip,
        sentinel.short_name,
        sentinel.frequency,
        check_level=sentinel.check_level,
    )
    mock_get_cmor_checker.return_value.assert_called_once_with(sentinel.cube)
    (
        mock_get_cmor_checker.return_value.return_value.check_data.
        assert_called_once_with()
    )
    assert cube == sentinel.checked_cube


def test_cmor_check(mocker):
    """Test ``cmor_check``"""
    mock_cmor_check_metadata = mocker.patch.object(
        esmvalcore.cmor.check,
        'cmor_check_metadata',
        autospec=True,
        return_value=sentinel.cube_after_check_metadata,
    )
    mock_cmor_check_data = mocker.patch.object(
        esmvalcore.cmor.check,
        'cmor_check_data',
        autospec=True,
        return_value=sentinel.cube_after_check_data,
    )

    cube = cmor_check(
        sentinel.cube,
        sentinel.cmor_table,
        sentinel.mip,
        sentinel.short_name,
        sentinel.frequency,
        sentinel.check_level,
    )

    mock_cmor_check_metadata.assert_called_once_with(
        sentinel.cube,
        sentinel.cmor_table,
        sentinel.mip,
        sentinel.short_name,
        sentinel.frequency,
        check_level=sentinel.check_level,
    )
    mock_cmor_check_data.assert_called_once_with(
        sentinel.cube_after_check_metadata,
        sentinel.cmor_table,
        sentinel.mip,
        sentinel.short_name,
        sentinel.frequency,
        check_level=sentinel.check_level,
    )
    assert cube == sentinel.cube_after_check_data
