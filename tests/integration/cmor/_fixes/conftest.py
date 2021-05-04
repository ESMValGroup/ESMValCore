"""Pytest configuration for ``cmor._fixes``."""

from pathlib import Path

import pytest


@pytest.fixture
def test_data_path():
    """Path to test data for CMOR fixes."""
    return Path(__file__).resolve().parent / 'test_data'
