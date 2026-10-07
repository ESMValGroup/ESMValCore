"""Integration tests for :mod:`esmvalcore.io.wdcc`."""

from __future__ import annotations

import importlib.resources
from typing import TYPE_CHECKING

import pytest
import requests.utils
import yaml

import esmvalcore.config
from esmvalcore.io import wdcc

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from pytest_mock import MockerFixture


def _create_data_source(
    project: str,
    download_dir: Path,
) -> wdcc.WDCCDataSource:
    """Create a data source from the example configuration."""
    config_file = (
        importlib.resources.files(esmvalcore.config)
        / "configurations"
        / "data-wdcc.yml"
    )
    cfg = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    kwargs = cfg["projects"][project]["data"]["wdcc"]
    kwargs.pop("type")
    kwargs["download_dir"] = download_dir
    return wdcc.WDCCDataSource(name="wdcc", project=project, **kwargs)


@pytest.fixture(autouse=True)
def cache_dir(tmp_path: Path, mocker: MockerFixture) -> Iterator[Path]:
    """Use a temporary cache directory."""
    cache_dir = tmp_path / "cache"
    mocker.patch.object(
        wdcc.platformdirs,
        "user_cache_path",
        return_value=cache_dir,
    )
    wdcc._get_cached_session.cache_clear()
    yield cache_dir
    wdcc._get_cached_session.cache_clear()


@pytest.fixture
def data_source(tmp_path: Path) -> wdcc.WDCCDataSource:
    """Create a CMIP5 data source from the example configuration."""
    return _create_data_source("CMIP5", tmp_path / "climate_data")


FACETS = {
    "project": "CMIP5",
    "product": ["output1", "output2"],
    "institute": "MPI-M",
    "dataset": "MPI-ESM-LR",
    "exp": "historical",
    "modeling_realm": "atmos",
    "ensemble": "r1i1p1",
}


@pytest.mark.online
def test_find_data(data_source: wdcc.WDCCDataSource) -> None:
    files = data_source.find_data(
        **FACETS,
        frequency="mon",
        mip="Amon",
        short_name="tas",
        timerange="2000/2005",
    )
    assert [f.name for f in files] == [
        "tas_Amon_MPI-ESM-LR_historical_r1i1p1_185001-200512.nc",
    ]
    assert files[0].facets["version"] == "v20120315"
    assert files[0].local_file == (
        data_source.download_dir
        / "cmip5/output1/MPI-M/MPI-ESM-LR/historical/mon/atmos/Amon/r1i1p1"
        / "v20120315"
        / files[0].name
    )


@pytest.mark.online
@pytest.mark.skipif(
    requests.utils.get_netrc_auth("https://www.wdc-climate.de") is None,
    reason="No WDCC credentials configured in ~/.netrc",
)
def test_download(data_source: wdcc.WDCCDataSource) -> None:
    files = data_source.find_data(
        **{**FACETS, "ensemble": "r0i0p0"},
        frequency="fx",
        mip="fx",
        short_name="sftlf",
    )
    assert len(files) == 1
    wdcc.download(files)
    cubes = files[0].to_iris()
    assert cubes[0].var_name == "sftlf"


@pytest.mark.online
def test_find_data_cordex(tmp_path: Path) -> None:
    cordex_source = _create_data_source("CORDEX", tmp_path / "climate_data")
    files = cordex_source.find_data(
        project="CORDEX",
        domain="EUR-11",
        institute="CLMcom",
        driver="CNRM-CERFACS-CNRM-CM5",
        exp="historical",
        ensemble="r1i1p1",
        dataset="CCLM4-8-17",
        rcm_version="v1",
        frequency="mon",
        mip="mon",
        short_name="tas",
        timerange="2000/2005",
    )
    assert [f.name for f in files] == [
        "tas_EUR-11_CNRM-CERFACS-CNRM-CM5_historical_r1i1p1_CLMcom-CCLM4-8-17_v1_mon_199101-200012.nc",
        "tas_EUR-11_CNRM-CERFACS-CNRM-CM5_historical_r1i1p1_CLMcom-CCLM4-8-17_v1_mon_200101-200512.nc",
    ]
    assert files[0].facets["version"] == "v20140515"
    assert files[0].facets["dataset"] == "CCLM4-8-17"
    assert files[0].local_file == (
        tmp_path
        / "climate_data"
        / "cordex/output/EUR-11/CLMcom/CNRM-CERFACS-CNRM-CM5/historical"
        / "r1i1p1/CCLM4-8-17/v1/mon/tas/v20140515"
        / files[0].name
    )
