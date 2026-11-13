"""Tests for the built-in configuration files of :mod:`esmvalcore.io.xarray`."""

from __future__ import annotations

import importlib.resources
import netrc
from typing import Any

import pytest
import yaml

import esmvalcore.config
from esmvalcore.io.xarray import XarrayDataSource

CONFIGURATIONS = (
    importlib.resources.files(esmvalcore.config) / "configurations"
)


def _has_netrc_entry(host: str) -> bool:
    try:
        return netrc.netrc().authenticators(host) is not None
    except (FileNotFoundError, netrc.NetrcParseError):
        return False


def load_data_source(filename: str) -> XarrayDataSource:
    """Create the single data source configured in ``filename``."""
    config = yaml.safe_load(
        (CONFIGURATIONS / filename).read_text(encoding="utf-8"),
    )
    ((project, project_config),) = config["projects"].items()
    ((name, kwargs),) = project_config["data"].items()
    kwargs = dict(kwargs)
    kwargs.pop("type")
    return XarrayDataSource(name=name, project=project, **kwargs)


EDH = pytest.mark.skipif(
    not _has_netrc_entry("data.earthdatahub.destine.eu"),
    reason="No Earth Data Hub token in ~/.netrc",
)


@pytest.mark.online
@pytest.mark.parametrize(
    ("filename", "facets"),
    [
        pytest.param(
            "data-gcs-era5.yml",
            {"short_name": "tas", "timerange": "20000101T00/20000101T00"},
            id="gcs-era5",
        ),
        pytest.param(
            "data-edh-era5.yml",
            {"short_name": "tas", "timerange": "200001/200001"},
            id="edh-era5",
            marks=EDH,
        ),
        pytest.param(
            "data-edh-cmip6.yml",
            {
                "short_name": "tas",
                "dataset": "CMCC-CM2-SR5",
                "exp": "ssp245",
                "timerange": "20150101/20150101",
            },
            id="edh-cmip6",
            marks=EDH,
        ),
    ],
)
def test_load_one_time_step(filename: str, facets: dict[str, Any]) -> None:
    data_source = load_data_source(filename)
    (data_element,) = data_source.find_data(**facets)
    cubes = data_element.to_iris()
    (cube,) = (c for c in cubes if c.ndim == 3)
    assert cube.shape[0] == 1
    assert cube.has_lazy_data()
    assert cube.core_data().max().compute() > 0
