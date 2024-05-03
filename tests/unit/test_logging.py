"""Tests for logging config submodule."""

import logging
from pathlib import Path

import pytest

from esmvalcore.config._logging import FilterMultipleNames, configure_logging


@pytest.mark.parametrize('level', (None, 'INFO', 'DEBUG'))
def test_logging_with_level(level):
    """Test log level configuration."""
    ret = configure_logging(console_log_level=level)
    assert isinstance(ret, list)
    assert len(ret) == 0

    root = logging.getLogger()

    assert len(root.handlers) == 1


def test_logging_with_output_dir(tmp_path):
    """Test that paths are configured."""
    ret = configure_logging(output_dir=tmp_path)
    assert isinstance(ret, list)
    for path in ret:
        assert tmp_path == Path(path).parent

    root = logging.getLogger()

    assert len(root.handlers) == len(ret) + 1


def test_logging_log_level_invalid():
    """Test failure condition for invalid level specification."""
    with pytest.raises(ValueError):
        configure_logging(console_log_level='FAIL')


@pytest.mark.parametrize(
    'names,mode,output',
    [
        (['test'], 'allow', False),
        (['test'], 'disallow', True),
        (['test', 'another.test'], 'allow', False),
        (['test', 'another.test'], 'disallow', True),
        (['test', 'm.a.b.c'], 'allow', False),
        (['test', 'm.a.b.c'], 'disallow', True),
        (['a.b.c'], 'allow', True),
        (['a.b.c'], 'disallow', False),
        (['a'], 'allow', True),
        (['a'], 'disallow', False),
        (['a.b', 'test'], 'allow', True),
        (['a.b', 'test'], 'disallow', False),
        (['a.b', 'a.b.c'], 'allow', True),
        (['a.b', 'a.b.c'], 'disallow', False),
    ]
)
def test_filter_multiple_names(names, mode, output):
    """Test `FilterMultipleNames`."""
    filter = FilterMultipleNames(names, mode)
    record = logging.LogRecord(
        'a.b.c', 'level', 'path', 'lineno', 'msg', [], 'exc_info'
    )
    assert filter.filter(record) is output
