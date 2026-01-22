"""Unit tests for esmvalcore.io.intake_esm."""

from __future__ import annotations

import importlib.resources
from typing import TYPE_CHECKING
from pathlib import Path
from importlib.resources import files as importlib_files

import iris.cube
import pandas as pd
import pytest
import xarray as xr
import yaml

import intake
import esmvalcore.io.intake_esm
from esmvalcore.io.intake_esm import IntakeEsmDataset, IntakeEsmDataSource

if TYPE_CHECKING:
    from pytest import MonkeyPatch
    from pytest_mock import MockerFixture

    from esmvalcore.config import Session


esm_ds_fhandle = (
    Path(importlib_files("tests"))
    / "sample_data"
    / "intake-esm"
    / "catalog"
    / "cmip6-netcdf.json"
)


def test_intakeesmdataset_repr() -> None:
    cat = intake.open_esm_datastore(esm_ds_fhandle.as_posix())
    dataset = IntakeEsmDataset(name="id", facets={}, catalog=cat)
    assert repr(dataset) == "IntakeEsmDataset(name='id')"


def test_prepare(mocker: MockerFixture) -> None:
    """IntakeEsmDataset.prepare should not do anything (just pass)."""
    cat = intake.open_esm_datastore(esm_ds_fhandle.as_posix())
    dataset = IntakeEsmDataset(name="id", facets={}, catalog=cat)

    # prepare() just passes for intake-esm, so we just verify it doesn't raise
    dataset.prepare()


def test_attributes_raises_before_to_iris() -> None:
    """Accessing attributes before to_iris should raise ValueError."""
    cat = intake.open_esm_datastore(esm_ds_fhandle.as_posix())
    dataset = IntakeEsmDataset(name="id", facets={}, catalog=cat)
    with pytest.raises(ValueError, match="Attributes have not been read yet"):
        _ = dataset.attributes


def test_to_iris(mocker: MockerFixture) -> None:
    """`to_iris` should load the data and cache attributes."""
    cat = intake.open_esm_datastore(esm_ds_fhandle.as_posix())
    key = "my.dataset.1"
    mocker.patch(
        "esmvalcore.io.intake_esm._to_path_dict",
        return_value={key: ["/path/to/file.nc"]},
    )
    ds = xr.Dataset(attrs={"attr": "value"})
    mocker.patch.object(cat, "to_dask", return_value=ds)

    cubes = mocker.sentinel.cubes
    mocker.patch.object(
        esmvalcore.io.intake_esm,
        "dataset_to_iris",
        return_value=cubes,
    )

    dataset = IntakeEsmDataset(name=key, facets={}, catalog=cat)
    result = dataset.to_iris()
    assert result is cubes

    assert dataset.attributes == {
        "attr": "value",
        "source_file": "/path/to/file.nc",
    }


@pytest.mark.online
@pytest.mark.skip(reason="Requires intake-esm catalog configuration")
def test_to_iris_online():
    """`to_iris` should load data from a real intake-esm catalog."""
    data_source = IntakeEsmDataSource(
        name="src",
        project="CMIP6",
        priority=1,
        facets={
            "activity": "activity_drs",
            "dataset": "source_id",
            "ensemble": "member_id",
            "exp": "experiment_id",
            "grid": "grid_label",
            "institute": "institution_id",
            "mip": "table_id",
            "project": "project",
            "short_name": "variable_id",
        },
        values={},
    )
    results = data_source.find_data(
        dataset="CanESM5",
        ensemble="r1i1p1f1",
        exp="historical",
        grid="gn",
        mip="fx",
        project="CMIP6",
        short_name="areacella",
    )
    assert len(results) == 1
    dataset = results[0]
    assert isinstance(dataset, IntakeEsmDataset)
    cubes = dataset.to_iris()
    assert len(cubes) == 1
    assert isinstance(cubes[0], iris.cube.Cube)
    # Check that the "source_file" attributes is present for debugging.
    assert "source_file" in dataset.attributes
    assert dataset.attributes["source_file"].endswith(".nc")


def test_find_data_no_results_sets_debug_info(mocker: MockerFixture) -> None:
    """When catalog.search returns empty results, find_data should return empty list and set debug_info."""
    data_source = IntakeEsmDataSource(
        name="src",
        project="CMIP6",
        priority=1,
        facets={"short_name": "variable_id"},
    )

    cat = intake.open_esm_datastore(esm_ds_fhandle.as_posix())
    # Mock search to return an empty result (intake-esm returns the catalog itself)
    empty_cat = intake.open_esm_datastore(esm_ds_fhandle.as_posix()).search(
        variable_id="bogus_variable"
    )
    mocker.patch.object(
        cat,
        "search",
        return_value=empty_cat,
    )
    mocker.patch.object(empty_cat, "__len__", return_value=0)
    data_source.catalog = cat

    result = data_source.find_data(short_name="tas")
    assert result == []
    expected_debug_info = "`intake_esm.esm_datastore().search(variable_id=['tas'])` did not return any results."
    assert data_source.debug_info == expected_debug_info


def test_find_data(mocker: MockerFixture, monkeypatch: MonkeyPatch) -> None:
    """find_data should convert catalog.df rows into IntakeEsmDataset instances."""
    cat = intake.open_esm_datastore(esm_ds_fhandle.as_posix())
    # Mock the project attribute
    mock_project = mocker.MagicMock()
    mock_project.master_id_facets.return_value = [
        "project",
        "activity_drs",
        "institution_id",
        "source_id",
        "experiment_id",
        "member_id",
        "table_id",
        "variable_id",
        "grid_label",
    ]
    cat.project = mock_project
    cat.df = pd.DataFrame.from_dict(
        {
            "project": ["CMIP6", "CMIP6"],
            "mip_era": ["CMIP6", "CMIP6"],
            "activity_drs": ["CMIP", "ScenarioMIP"],
            "institution_id": ["CCCma", "CCCma"],
            "source_id": ["CanESM5", "CanESM5"],
            "experiment_id": ["historical", "ssp585"],
            "member_id": ["r1i1p1f1", "r1i1p1f1"],
            "table_id": ["Amon", "Amon"],
            "variable_id": ["tas", "tas"],
            "grid_label": ["gn", "gn"],
            "version": ["20190429", "20190429"],
            "id": [
                [
                    "CMIP6.CMIP.CCCma.CanESM5.historical.r1i1p1f1.Amon.tas.gn.v20190429|crd-esgf-drc.ec.gc.ca",
                    "CMIP6.CMIP.CCCma.CanESM5.historical.r1i1p1f1.Amon.tas.gn.v20190429|eagle.alcf.anl.gov",
                    "CMIP6.CMIP.CCCma.CanESM5.historical.r1i1p1f1.Amon.tas.gn.v20190429|esgf-data04.diasjp.net",
                    "CMIP6.CMIP.CCCma.CanESM5.historical.r1i1p1f1.Amon.tas.gn.v20190429|esgf-node.ornl.gov",
                ],
                [
                    "CMIP6.ScenarioMIP.CCCma.CanESM5.ssp585.r1i1p1f1.Amon.tas.gn.v20190429|crd-esgf-drc.ec.gc.ca",
                    "CMIP6.ScenarioMIP.CCCma.CanESM5.ssp585.r1i1p1f1.Amon.tas.gn.v20190429|eagle.alcf.anl.gov",
                    "CMIP6.ScenarioMIP.CCCma.CanESM5.ssp585.r1i1p1f1.Amon.tas.gn.v20190429|esgf-data04.diasjp.net",
                    "CMIP6.ScenarioMIP.CCCma.CanESM5.ssp585.r1i1p1f1.Amon.tas.gn.v20190429|esgf-node.ornl.gov",
                ],
            ],
        },
    )

    # Patch search to return the catalog itself with the df we set
    mocker.patch.object(cat, "search", return_value=cat)
    mocker.patch.object(cat, "__len__", return_value=2)

    data_source = IntakeEsmDataSource(
        name="src",
        project="CMIP6",
        priority=1,
        facets={
            "activity": "activity_drs",
            "dataset": "source_id",
            "ensemble": "member_id",
            "exp": "experiment_id",
            "institute": "institution_id",
            "grid": "grid_label",
            "mip": "table_id",
            "project": "project",
            "short_name": "variable_id",
        },
        values={},
    )
    data_source.catalog = cat

    # Call find_data - it should use the df we set and return two datasets
    results = data_source.find_data(short_name="tas")
    assert isinstance(results, list)
    assert len(results) == 2

    dataset = results[0]
    assert isinstance(dataset, IntakeEsmDataset)
    assert dataset.name == "CMIP6.CMIP.CCCma.CanESM5.historical.r1i1p1f1.Amon.tas.gn"
    assert hash(dataset) == hash((dataset.name, "v20190429"))

    assert dataset.facets == {
        "activity": "CMIP",
        "dataset": "CanESM5",
        "ensemble": "r1i1p1f1",
        "exp": "historical",
        "grid": "gn",
        "institute": "CCCma",
        "mip": "Amon",
        "project": "CMIP6",
        "short_name": "tas",
        "version": "v20190429",
    }
    dataset = results[1]
    assert dataset.name == "CMIP6.ScenarioMIP.CCCma.CanESM5.ssp585.r1i1p1f1.Amon.tas.gn"
    assert dataset.facets == {
        "activity": "ScenarioMIP",
        "dataset": "CanESM5",
        "ensemble": "r1i1p1f1",
        "exp": "ssp585",
        "grid": "gn",
        "institute": "CCCma",
        "mip": "Amon",
        "project": "CMIP6",
        "short_name": "tas",
        "version": "v20190429",
    }


@pytest.fixture
def data_sources(session: Session) -> list[esmvalcore.io.protocol.DataSource]:
    """Fixture providing the default list of IntakeEsmDataSource data sources."""
    with importlib.resources.as_file(
        importlib.resources.files(esmvalcore.config)
        / "configurations"
        / "data-intake-esm.yml",
    ) as config_file:
        cfg = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    session["projects"] = cfg["projects"]
    return esmvalcore.io.load_data_sources(session)


@pytest.mark.online
@pytest.mark.skip(reason="Requires intake-esm catalog configuration")
@pytest.mark.parametrize(
    ("facets", "expected_names"),
    [
        pytest.param(
            {
                "dataset": "CanESM5",
                "ensemble": "r1i1p1f1",
                "exp": ["historical", "ssp585"],
                "grid": "gn",
                "mip": "Amon",
                "project": "CMIP6",
                "short_name": "tas",
                "timerange": "1850/2100",
            },
            {
                "CMIP6.CMIP.CCCma.CanESM5.historical.r1i1p1f1.Amon.tas.gn",
                "CMIP6.ScenarioMIP.CCCma.CanESM5.ssp585.r1i1p1f1.Amon.tas.gn",
            },
            id="CMIP6",
        ),
        pytest.param(
            {
                "dataset": "CanESM5",
                "ensemble": "r[1-3]i1p1f1",
                "exp": "historical",
                "grid": "gn",
                "mip": "Amon",
                "project": "CMIP6",
                "short_name": "tas",
                "timerange": "1850/2100",
            },
            {
                "CMIP6.CMIP.CCCma.CanESM5.historical.r1i1p1f1.Amon.tas.gn",
                "CMIP6.CMIP.CCCma.CanESM5.historical.r2i1p1f1.Amon.tas.gn",
                "CMIP6.CMIP.CCCma.CanESM5.historical.r3i1p1f1.Amon.tas.gn",
            },
            id="CMIP6-with-glob-pattern",
        ),
        pytest.param(
            {
                "dataset": "ACCESS1-0",
                "ensemble": "r1i1p1",
                "exp": ["historical", "rcp85"],
                "mip": "Amon",
                "project": "CMIP5",
                "short_name": "tas",
            },
            {
                "CSIRO-BOM.ACCESS1.0.historical.mon.atmos.Amon.r1i1p1.tas",
                "CSIRO-BOM.ACCESS1.0.rcp85.mon.atmos.Amon.r1i1p1.tas",
            },
            id="CMIP5",
        ),
        pytest.param(
            {
                "dataset": "cccma_cgcm3_1",
                "ensemble": "run1",
                "exp": "historical",
                "mip": "A1",
                "project": "CMIP3",
                "short_name": "tas",
            },
            {
                "CMIP3.CCCMA.cccma_cgcm3_1.historical.day.atmos.run1.tas",
                "CMIP3.CCCMA.cccma_cgcm3_1.historical.mon.atmos.run1.tas",
            },
            id="CMIP3",
        ),
        pytest.param(
            {
                "dataset": "ERA-5",
                "project": "obs4MIPs",
                "short_name": "tas",
            },
            {
                "obs4MIPs.ECMWF.ERA-5.mon.tas.gn",
            },
            id="obs4MIPs",
        ),
    ],
)
def test_find_data_online(
    data_sources: list[IntakeEsmDataSource],
    facets: dict[str, str | list[str]],
    expected_names: list[str],
) -> None:
    """Test finding data from a real intake-esm catalog."""
    data_source = next(ds for ds in data_sources if ds.project == facets["project"])
    result = data_source.find_data(**facets)
    assert len(result) > 0
    result_names = {ds.name for ds in result}
    assert result_names == expected_names
