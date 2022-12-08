"""Tests for logging config submodule."""

import logging
from pathlib import Path

import pytest

from esmvalcore._config._logging import configure_logging


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
