"""Tests for :func:`esmvalcore.io.load_data_sources`."""

import importlib.resources
from dataclasses import dataclass

import pytest

import esmvalcore.config
import esmvalcore.io


def test_configurations_valid(cfg_default: esmvalcore.config.Config) -> None:
    """Test that the data sources configuration in esmvalcore/config/configurations are valid."""
    configurations = (
        importlib.resources.files(esmvalcore.config) / "configurations"
    )
    with importlib.resources.as_file(configurations) as config_dir:
        cfg_default.load_from_dirs([config_dir])
    session = cfg_default.start_session("test")
    data_sources = esmvalcore.io.load_data_sources(session)
    for data_source in data_sources:
        assert isinstance(data_source, esmvalcore.io.DataSource)


def test_load_data_sources_unknown_project(
    session: esmvalcore.config.Session,
) -> None:
    """Test that loading data sources for an unknown project raises."""
    with pytest.raises(ValueError, match=r"Unknown project 'unknown'.*"):
        esmvalcore.io.load_data_sources(session, project="unknown")


def test_load_data_sources_no_data_sources_configured(
    session: esmvalcore.config.Session,
) -> None:
    """Test that loading data sources when no data sources are configured raises."""
    session["projects"].clear()
    with pytest.raises(
        ValueError,
        match=r"No data sources found. Check your configuration under 'projects'",
    ):
        esmvalcore.io.load_data_sources(session)


def test_load_data_sources_no_project_data_sources_configured(
    session: esmvalcore.config.Session,
) -> None:
    """Test that loading data sources when no data sources are configured raises."""
    session["projects"]["test"] = {}
    with pytest.raises(
        ValueError,
        match=r"No data sources found for project 'test'.*",
    ):
        esmvalcore.io.load_data_sources(session, project="test")


@dataclass
class IncompleteDataSource:
    """An incomplete data source class for testing."""

    name: str
    project: str
    priority: int
    # Note the missing implementation of DataSource methods.


def test_load_data_sources_invalid_data_source_type(
    session: esmvalcore.config.Session,
) -> None:
    """Test that loading data sources with an invalid data source type raises."""
    session["projects"]["test"] = {
        "data": {
            "invalid_source": {
                "type": "tests.unit.io.test_load_data_sources.IncompleteDataSource",
            },
        },
    }
    with pytest.raises(
        TypeError,
        match=r"Expected a data source of type `esmvalcore.io.protocol.DataSource`.*",
    ):
        esmvalcore.io.load_data_sources(session, project="test")
