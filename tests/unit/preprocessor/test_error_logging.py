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
    assert mock_logger.debug.call_args.kwargs == {}
    debug_call_args = mock_logger.debug.call_args.args
    assert debug_call_args[0] == ("Running %s with option(s)\n%s\non argument"
                                  "(s)\n%s%s")
    assert debug_call_args[1] == "failing_function"
    assert debug_call_args[2] == "{'test': 42}"
    if isinstance(items, (PreprocessorFile, Cube, str)):
        assert debug_call_args[3] == repr(items)
    else:
        for item in items:
            assert repr(item) in debug_call_args[3]


def assert_error_call_ok(mock_logger):
    """Check error call."""
    mock_logger.error.assert_called_once()
    assert mock_logger.error.call_args.kwargs == {}
    error_call_args = mock_logger.error.call_args.args
    assert error_call_args[0] == "Failed to run %s"
    expected_str = ("failing_function with option(s)\n{'test': 42}\non "
                    "argument(s)\n")
    assert expected_str in error_call_args[1]


KWARGS = {'test': 42}
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
    assert mock_logger.debug.call_args.args[4] == ""

    # Error call
    assert_error_call_ok(mock_logger)
    error_call_args = mock_logger.error.call_args.args
    if isinstance(items, (PreprocessorFile, Cube, str)):
        assert repr(items) in error_call_args[1]
    else:
        for item in items:
            assert repr(item) in error_call_args[1]
    assert "\nOriginal input file(s):\n" not in error_call_args[1]
    assert "further file(s) not shown here;" not in error_call_args[1]
    assert "further argument(s) not shown here;" not in error_call_args[1]


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
    assert mock_logger.debug.call_args.args[4] == (
        "\nOriginal input file(s):\n['x', 'y', 'z', 'w']")

    # Error call
    assert_error_call_ok(mock_logger)
    error_call_args = mock_logger.error.call_args.args
    if isinstance(items, (PreprocessorFile, Cube, str)):
        assert repr(items) in error_call_args[1]
    else:
        for item in items:
            assert repr(item) in error_call_args[1]
    assert ("\nOriginal input file(s):\n['x', 'y', 'z', 'w']" in
            error_call_args[1])
    assert "further file(s) not shown here;" not in error_call_args[1]
    assert "further argument(s) not shown here;" not in error_call_args[1]


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
    assert mock_logger.debug.call_args.args[4] == (
        "\nOriginal input file(s):\n['x', 'y', 'z', 'w', 'v', 'u']")

    # Error call
    assert_error_call_ok(mock_logger)
    error_call_args = mock_logger.error.call_args.args
    if isinstance(items, (PreprocessorFile, Cube, str)):
        assert repr(items) in error_call_args[1]
    else:
        for item in items:
            assert repr(item) in error_call_args[1]
    assert ("\nOriginal input file(s):\n['x', 'y', 'z', 'w']\n" in
            error_call_args[1])
    assert "['x', 'y', 'z', 'w', 'v', 'u']" not in error_call_args[1]
    assert "\n(and 2 further file(s) not shown here;" in error_call_args[1]
    assert "further argument(s) not shown here;" not in error_call_args[1]


@pytest.mark.parametrize('items', TEST_ITEMS_LONG)
@mock.patch('esmvalcore.preprocessor.logger', autospec=True)
def test_long_items_no_input_files(mock_logger, items):
    """Test long list of items and no input files."""
    with pytest.raises(ValueError, match=VALUE_ERROR_MSG):
        _run_preproc_function(failing_function, items, KWARGS)
    assert len(mock_logger.mock_calls) == 2

    # Debug call
    assert_debug_call_ok(mock_logger, items)
    assert mock_logger.debug.call_args.args[4] == ""

    # Error call
    assert_error_call_ok(mock_logger)
    error_call_args = mock_logger.error.call_args.args
    items = list(items)
    for item in items[:4]:
        assert repr(item) in error_call_args[1]
    for item in items[4:]:
        assert repr(item) not in error_call_args[1]
    assert "\nOriginal input file(s):\n" not in error_call_args[1]
    assert "further file(s) not shown here;" not in error_call_args[1]
    assert "\n(and 2 further argument(s) not shown here;" in error_call_args[1]


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
    assert mock_logger.debug.call_args.args[4] == (
        "\nOriginal input file(s):\n['x', 'y', 'z', 'w']")

    # Error call
    assert_error_call_ok(mock_logger)
    error_call_args = mock_logger.error.call_args.args
    items = list(items)
    for item in items[:4]:
        assert repr(item) in error_call_args[1]
    for item in items[4:]:
        assert repr(item) not in error_call_args[1]
    assert ("\nOriginal input file(s):\n['x', 'y', 'z', 'w']" in
            error_call_args[1])
    assert "further file(s) not shown here;" not in error_call_args[1]
    assert "\n(and 2 further argument(s) not shown here;" in error_call_args[1]


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
    assert mock_logger.debug.call_args.args[4] == (
        "\nOriginal input file(s):\n['x', 'y', 'z', 'w', 'v', 'u']")

    # Error call
    assert_error_call_ok(mock_logger)
    error_call_args = mock_logger.error.call_args.args
    items = list(items)
    for item in items[:4]:
        assert repr(item) in error_call_args[1]
    for item in items[4:]:
        assert repr(item) not in error_call_args[1]
    assert ("\nOriginal input file(s):\n['x', 'y', 'z', 'w']\n" in
            error_call_args[1])
    assert "['x', 'y', 'z', 'w', 'v', 'u']" not in error_call_args[1]
    assert "\n(and 2 further file(s) not shown here;" in error_call_args[1]
    assert "\n(and 2 further argument(s) not shown here;" in error_call_args[1]


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
