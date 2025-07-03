"""Unit tests for esmvalcore.io.intake_esgf."""

import importlib.resources

import intake_esgf
import iris.cube
import pandas as pd
import pytest
import xarray as xr
import yaml
from pytest import MonkeyPatch
from pytest_mock import MockerFixture

import esmvalcore.io.intake_esgf
from esmvalcore.config import Session
from esmvalcore.io.intake_esgf import IntakeESGFDataset, IntakeESGFDataSource


def test_prepare(mocker: MockerFixture) -> None:
    """IntakeESGFDataset.prepare should call the catalog.to_path_dict method."""
    cat = intake_esgf.ESGFCatalog()
    to_path_mock = mocker.patch.object(cat, "to_path_dict", autospec=True)
    dataset = IntakeESGFDataset(name="id", facets={}, catalog=cat)

    dataset.prepare()
    to_path_mock.assert_called_once_with()


def test_attributes_raises_before_to_iris() -> None:
    """Accessing attributes before to_iris should raise ValueError."""
    cat = intake_esgf.ESGFCatalog()
    dataset = IntakeESGFDataset(name="id", facets={}, catalog=cat)
    with pytest.raises(ValueError, match="Attributes have not been read yet"):
        _ = dataset.attributes


def test_to_iris(mocker: MockerFixture) -> None:
    """`to_iris` should load the data and cache attributes."""
    cat = intake_esgf.ESGFCatalog()
    key = "my.dataset.1"
    mocker.patch.object(
        cat,
        "to_path_dict",
        return_value={key: ["/path/to/file.nc"]},
    )
    ds = xr.Dataset(attrs={"attr": "value"})
    mocker.patch.object(cat, "to_dataset_dict", return_value={key: ds})

    cubes = mocker.sentinel.cubes
    mocker.patch.object(
        esmvalcore.io.intake_esgf,
        "dataset_to_iris",
        return_value=cubes,
    )

    dataset = IntakeESGFDataset(name=key, facets={}, catalog=cat)
    result = dataset.to_iris(ignore_warnings=[{"message": "ignore"}])
    assert result is cubes

    assert dataset.attributes == {
        "attr": "value",
        "source_file": "/path/to/file.nc",
    }


@pytest.mark.online
def test_to_iris_online():
    """`to_iris` should load data from a real ESGF catalog."""
    data_source = IntakeESGFDataSource(
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
    assert isinstance(dataset, IntakeESGFDataset)
    cubes = dataset.to_iris()
    assert len(cubes) == 1
    assert isinstance(cubes[0], iris.cube.Cube)
    # Check that the "source_file" attributes is present for debugging.
    assert "source_file" in dataset.attributes
    assert dataset.attributes["source_file"].endswith(".nc")


def test_find_data_no_results_sets_debug_info(mocker: MockerFixture) -> None:
    """When catalog.search raises NoSearchResults, find_data should return empty list and set debug_info."""
    data_source = IntakeESGFDataSource(
        name="src",
        project="CMIP6",
        priority=1,
        facets={"short_name": "variable_id"},
    )

    cat = intake_esgf.ESGFCatalog()
    # Ensure last_search is present so debug_info can be constructed
    cat.last_search = {"variable_id": "tas"}
    mocker.patch.object(
        cat,
        "search",
        side_effect=intake_esgf.exceptions.NoSearchResults("no results"),
    )
    data_source.catalog = cat

    result = data_source.find_data(short_name="tas")
    assert result == []
    expected_debug_info = "intake_esgf.ESGFCatalog.search(variable_id=['tas']) did not return any results."
    assert data_source.debug_info == expected_debug_info


def test_find_data(mocker: MockerFixture, monkeypatch: MonkeyPatch):
    """find_data should convert catalog.df rows into IntakeESGFDataset instances."""
    cat = intake_esgf.ESGFCatalog()
    cat.project = intake_esgf.projects.projects["cmip6"]
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

    # Patch search to just record last_search
    def fake_search(**kwargs):
        cat.last_search = kwargs

    mocker.patch.object(cat, "search", side_effect=fake_search)

    data_source = IntakeESGFDataSource(
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

    # Call find_data - it should use the df we set and return one dataset
    results = data_source.find_data(short_name="tas")
    assert isinstance(results, list)
    assert len(results) == 2

    dataset = results[0]
    assert isinstance(dataset, IntakeESGFDataset)
    assert (
        dataset.name
        == "CMIP6.CMIP.CCCma.CanESM5.historical.r1i1p1f1.Amon.tas.gn"
    )

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
    }
    dataset = results[1]
    assert (
        dataset.name
        == "CMIP6.ScenarioMIP.CCCma.CanESM5.ssp585.r1i1p1f1.Amon.tas.gn"
    )
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
    }


@pytest.fixture
def data_sources(session: Session) -> list[esmvalcore.io.protocol.DataSource]:
    """Fixture providing the default list of IntakeESGFDataSource data sources."""
    with importlib.resources.as_file(
        importlib.resources.files(esmvalcore.config)
        / "configurations"
        / "intake-esgf.yml",
    ) as config_file:
        cfg = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    session["projects"] = cfg["projects"]
    return esmvalcore.io.load_data_sources(session)


@pytest.mark.online
@pytest.mark.parametrize(
    ("facets", "expected_names"),
    [
        (
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
        ),
        (
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
        ),
        (
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
        ),
        (
            {
                "dataset": "ERA-5",
                "project": "obs4MIPs",
                "short_name": "tas",
            },
            {
                "obs4MIPs.ECMWF.ERA-5.mon.tas.gn",
            },
        ),
    ],
)
def test_find_data_online(
    data_sources: list[IntakeESGFDataSource],
    facets: dict[str, str | list[str]],
    expected_names: list[str],
) -> None:
    """Test finding data from a real ESGF catalog."""
    data_source = next(
        ds for ds in data_sources if ds.project == facets["project"]
    )
    result = data_source.find_data(**facets)
    result_names = {ds.name for ds in result}
    assert result_names == expected_names
