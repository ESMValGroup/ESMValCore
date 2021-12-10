"""Unit tests for error logging of :mod:`esmvalcore.preprocessor`."""

from unittest import mock

import pytest
from iris.cube import Cube, CubeList

from esmvalcore.preprocessor import PreprocessorFile, _run_preproc_function

VALUE_ERROR_MSG = "This function is expected to fail with exactly this error"


def failing_function(*_, **__):
    """Raise ValueError."""
    raise ValueError(VALUE_ERROR_MSG)


def assert_debug_call_ok(mock_logger, items):
    """Check debug call."""
    mock_logger.debug.assert_called_once()
    assert mock_logger.debug.call_args[1] == {}
    debug_call_args = mock_logger.debug.call_args[0]
    assert debug_call_args[0] == (
        "Running preprocessor function '%s' on the data\n%s%s\nwith function "
        "argument(s)\n%s")
    assert debug_call_args[1] == "failing_function"
    if isinstance(items, (PreprocessorFile, Cube, str)):
        assert debug_call_args[2] == repr(items)
    else:
        for item in items:
            assert repr(item) in debug_call_args[2]
    assert debug_call_args[4] == "test = 42,\nlist = ['a', 'b']"


def assert_error_call_ok(mock_logger):
    """Check error call."""
    mock_logger.error.assert_called_once()
    assert mock_logger.error.call_args[1] == {}
    error_call_args = mock_logger.error.call_args[0]
    assert error_call_args[0] == (
        "Failed to run preprocessor function '%s' on the data\n%s%s\nwith "
        "function argument(s)\n%s")
    assert error_call_args[1] == "failing_function"
    assert error_call_args[4] == "test = 42,\nlist = ['a', 'b']"


KWARGS = {'test': 42, 'list': ['a', 'b']}
PREPROC_FILE = PreprocessorFile({'filename': 'a'}, {})
TEST_ITEMS_SHORT = [
    # Scalars
    PREPROC_FILE,
    Cube(0),
    'a',
    # 1-element lists
    [PREPROC_FILE],
    [Cube(0)],
    ['a'],
    # 1-element sets
    set([PREPROC_FILE]),
    set([Cube(0)]),
    set(['a']),
    # 1-element CubeList
    CubeList([Cube(0)]),
    # 4-element lists
    [PREPROC_FILE] * 4,
    [Cube(0)] * 4,
    ['a'] * 4,
    # 4-element sets
    set(['a', 'b', 'c', 'd']),
    # 4-element CubeList
    CubeList([Cube(0), Cube(1), Cube(2), Cube(3)]),
]
TEST_ITEMS_LONG = [
    # 6-element list
    ['a', 'b', 'c', 'd', 'e', 'f'],
    # 6-element set
    set(['a', 'b', 'c', 'd', 'e', 'f']),
]
SHORT_INPUT_FILES = ['x', 'y', 'z', 'w']
LONG_INPUT_FILES = ['x', 'y', 'z', 'w', 'v', 'u']


@pytest.mark.parametrize('items', TEST_ITEMS_SHORT)
@mock.patch('esmvalcore.preprocessor.logger', autospec=True)
def test_short_items_no_input_files(mock_logger, items):
    """Test short list of items and no input files."""
    with pytest.raises(ValueError, match=VALUE_ERROR_MSG):
        _run_preproc_function(failing_function, items, KWARGS)
    assert len(mock_logger.mock_calls) == 2

    # Debug call
    assert_debug_call_ok(mock_logger, items)
    assert mock_logger.debug.call_args[0][3] == ""

    # Error call
    assert_error_call_ok(mock_logger)
    error_call_args = mock_logger.error.call_args[0]
    if isinstance(items, (PreprocessorFile, Cube, str)):
        assert repr(items) in error_call_args[2]
    else:
        for item in items:
            assert repr(item) in error_call_args[2]
    assert "further argument(s) not shown here;" not in error_call_args[2]
    assert error_call_args[3] == ""


@pytest.mark.parametrize('items', TEST_ITEMS_SHORT)
@mock.patch('esmvalcore.preprocessor.logger', autospec=True)
def test_short_items_short_input_files(mock_logger, items):
    """Test short list of items and short list of input files."""
    with pytest.raises(ValueError, match=VALUE_ERROR_MSG):
        _run_preproc_function(failing_function, items, KWARGS,
                              input_files=SHORT_INPUT_FILES)
    assert len(mock_logger.mock_calls) == 2

    # Debug call
    assert_debug_call_ok(mock_logger, items)
    assert mock_logger.debug.call_args[0][3] == (
        "\nloaded from original input file(s)\n['x', 'y', 'z', 'w']")

    # Error call
    assert_error_call_ok(mock_logger)
    error_call_args = mock_logger.error.call_args[0]
    if isinstance(items, (PreprocessorFile, Cube, str)):
        assert repr(items) in error_call_args[2]
    else:
        for item in items:
            assert repr(item) in error_call_args[2]
    assert "further argument(s) not shown here;" not in error_call_args[2]
    assert error_call_args[3] == (
        "\nloaded from original input file(s)\n['x', 'y', 'z', 'w']")


@pytest.mark.parametrize('items', TEST_ITEMS_SHORT)
@mock.patch('esmvalcore.preprocessor.logger', autospec=True)
def test_short_items_long_input_files(mock_logger, items):
    """Test short list of items and long list of input files."""
    with pytest.raises(ValueError, match=VALUE_ERROR_MSG):
        _run_preproc_function(failing_function, items, KWARGS,
                              input_files=LONG_INPUT_FILES)
    assert len(mock_logger.mock_calls) == 2

    # Debug call
    assert_debug_call_ok(mock_logger, items)
    assert mock_logger.debug.call_args[0][3] == (
        "\nloaded from original input file(s)\n['x', 'y', 'z', 'w', 'v', 'u']")

    # Error call
    assert_error_call_ok(mock_logger)
    error_call_args = mock_logger.error.call_args[0]
    if isinstance(items, (PreprocessorFile, Cube, str)):
        assert repr(items) in error_call_args[2]
    else:
        for item in items:
            assert repr(item) in error_call_args[2]
    assert "further argument(s) not shown here;" not in error_call_args[2]
    assert error_call_args[3] == (
        "\nloaded from original input file(s)\n['x', 'y', 'z', 'w']\n(and 2 "
        "further file(s) not shown here; refer to the debug log for a full "
        "list)")


@pytest.mark.parametrize('items', TEST_ITEMS_LONG)
@mock.patch('esmvalcore.preprocessor.logger', autospec=True)
def test_long_items_no_input_files(mock_logger, items):
    """Test long list of items and no input files."""
    with pytest.raises(ValueError, match=VALUE_ERROR_MSG):
        _run_preproc_function(failing_function, items, KWARGS)
    assert len(mock_logger.mock_calls) == 2

    # Debug call
    assert_debug_call_ok(mock_logger, items)
    assert mock_logger.debug.call_args[0][3] == ""

    # Error call
    assert_error_call_ok(mock_logger)
    error_call_args = mock_logger.error.call_args[0]
    items = list(items)
    for item in items[:4]:
        assert repr(item) in error_call_args[2]
    for item in items[4:]:
        assert repr(item) not in error_call_args[2]
    assert "\n(and 2 further argument(s) not shown here;" in error_call_args[2]
    assert error_call_args[3] == ""


@pytest.mark.parametrize('items', TEST_ITEMS_LONG)
@mock.patch('esmvalcore.preprocessor.logger', autospec=True)
def test_long_items_short_input_files(mock_logger, items):
    """Test long list of items and short list of input files."""
    with pytest.raises(ValueError, match=VALUE_ERROR_MSG):
        _run_preproc_function(failing_function, items, KWARGS,
                              input_files=SHORT_INPUT_FILES)
    assert len(mock_logger.mock_calls) == 2

    # Debug call
    assert_debug_call_ok(mock_logger, items)
    assert mock_logger.debug.call_args[0][3] == (
        "\nloaded from original input file(s)\n['x', 'y', 'z', 'w']")

    # Error call
    assert_error_call_ok(mock_logger)
    error_call_args = mock_logger.error.call_args[0]
    items = list(items)
    for item in items[:4]:
        assert repr(item) in error_call_args[2]
    for item in items[4:]:
        assert repr(item) not in error_call_args[2]
    assert "\n(and 2 further argument(s) not shown here;" in error_call_args[2]
    assert error_call_args[3] == (
        "\nloaded from original input file(s)\n['x', 'y', 'z', 'w']")


@pytest.mark.parametrize('items', TEST_ITEMS_LONG)
@mock.patch('esmvalcore.preprocessor.logger', autospec=True)
def test_long_items_long_input_files(mock_logger, items):
    """Test long list of items and long list of input files."""
    with pytest.raises(ValueError, match=VALUE_ERROR_MSG):
        _run_preproc_function(failing_function, items, KWARGS,
                              input_files=LONG_INPUT_FILES)
    assert len(mock_logger.mock_calls) == 2

    # Debug call
    assert_debug_call_ok(mock_logger, items)
    assert mock_logger.debug.call_args[0][3] == (
        "\nloaded from original input file(s)\n['x', 'y', 'z', 'w', 'v', 'u']")

    # Error call
    assert_error_call_ok(mock_logger)
    error_call_args = mock_logger.error.call_args[0]
    items = list(items)
    for item in items[:4]:
        assert repr(item) in error_call_args[2]
    for item in items[4:]:
        assert repr(item) not in error_call_args[2]
    assert "\n(and 2 further argument(s) not shown here;" in error_call_args[2]
    assert error_call_args[3] == (
        "\nloaded from original input file(s)\n['x', 'y', 'z', 'w']\n(and 2 "
        "further file(s) not shown here; refer to the debug log for a full "
        "list)")


class MockAncestor():
    """Mock class for ancestors."""

    def __init__(self, filename):
        """Initialize mock ancestor."""
        self.filename = filename


def test_input_files_for_log():
    """Test :meth:`PreprocessorFile._input_files_for_log`."""
    ancestors = [
        MockAncestor('a.nc'),
        MockAncestor('b.nc'),
    ]
    preproc_file = PreprocessorFile({'filename': 'p.nc'}, {},
                                    ancestors=ancestors)

    assert preproc_file._input_files == ['a.nc', 'b.nc']
    assert preproc_file.files == ['a.nc', 'b.nc']
    assert preproc_file._input_files_for_log() is None

    preproc_file.files = ['c.nc', 'd.nc']
    assert preproc_file._input_files == ['a.nc', 'b.nc']
    assert preproc_file.files == ['c.nc', 'd.nc']
    assert preproc_file._input_files_for_log() == ['a.nc', 'b.nc']
