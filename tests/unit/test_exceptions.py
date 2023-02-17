import sys
from unittest import mock

import pytest

import esmvalcore.exceptions
from esmvalcore.exceptions import (
    ESMValCoreDeprecationWarning,
    SuppressedError,
    show_esmvalcore_deprecation_warnings,
)


@pytest.mark.parametrize('exception', [SuppressedError, ValueError])
def test_suppressedhook(capsys, exception):
    try:
        raise exception('error')
    except exception:
        args = sys.exc_info()
    sys.excepthook(*args)
    msg = capsys.readouterr().err
    if issubclass(exception, SuppressedError):
        assert msg == "SuppressedError: error\n"
    else:
        ending = "ValueError: error\n"
        assert msg.endswith(ending)
        # because `msg` also contains the traceback, it should be
        # longer than `ending`
        assert len(msg) > len(ending)


@pytest.fixture
def temp_esmvalcore_deprec_warnings():
    """Make sure to not overwrite ESMVALCORE_DEPRECATION_WARNINGS."""
    esmvalcore.exceptions.ESMVALCORE_DEPRECATION_WARNINGS = set()
    yield esmvalcore.exceptions.ESMVALCORE_DEPRECATION_WARNINGS
    esmvalcore.exceptions.ESMVALCORE_DEPRECATION_WARNINGS = set()


def test_esmvalcore_deprecation_warning(temp_esmvalcore_deprec_warnings):
    """Test that ESMValCoreDeprecationWarnings are stored correctly."""
    msg_1 = 'This is a test message'
    msg_2 = 'This is another test message'
    ESMValCoreDeprecationWarning(msg_1)
    ESMValCoreDeprecationWarning(msg_1)  # make sure that msg_1 is stored once
    ESMValCoreDeprecationWarning(msg_2)

    assert len(temp_esmvalcore_deprec_warnings) == 2
    assert msg_1 in temp_esmvalcore_deprec_warnings
    assert msg_2 in temp_esmvalcore_deprec_warnings


@mock.patch('esmvalcore.exceptions.logger', autospec=True)
def test_show_esmvalcore_deprecation_warnings(
    mock_logger,
    temp_esmvalcore_deprec_warnings,
):
    """Test show_esmvalcore_deprecation_warnings."""
    msg_1 = '999 This is test message'
    msg_2 = '000 This is another test message'
    ESMValCoreDeprecationWarning(msg_1)
    ESMValCoreDeprecationWarning(msg_1)  # make sure that msg_1 is stored once
    ESMValCoreDeprecationWarning(msg_2)

    show_esmvalcore_deprecation_warnings()

    assert mock_logger.warning.call_count == 3
    assert mock_logger.warning.mock_calls == [
        mock.call('Please consider the following ESMValCore deprecation '
                  'messages:'),
        mock.call(msg_2),  # messages are sorted alphabetically
        mock.call(msg_1),
    ]


@mock.patch('esmvalcore.exceptions.logger', autospec=True)
def test_show_esmvalcore_deprecation_warnings_no_warning(
    mock_logger,
    temp_esmvalcore_deprec_warnings,
):
    """Test show_esmvalcore_deprecation_warnings with no message."""
    show_esmvalcore_deprecation_warnings()

    mock_logger.warning.assert_not_called()
