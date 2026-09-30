"""Exercise the intake-esm source with a small catalog and real data."""

from __future__ import annotations

import csv
import importlib.resources
import json
import shutil
from typing import TYPE_CHECKING

import intake
import numpy as np
import pytest
import xarray as xr
import yaml

import esmvalcore.io
from esmvalcore._recipe import recipe as recipe_module
from esmvalcore.dataset import Dataset
from esmvalcore.io.intake_esm import IntakeEsmDataSource

if TYPE_CHECKING:
    from pathlib import Path

    from esmvalcore.config import Session


@pytest.fixture
def small_catalog(tmp_path: Path) -> Path:
    """Create an intake-esm catalog containing one small NetCDF file."""
    data_path = tmp_path / "tas.nc"
    xr.Dataset(
        {
            "tas": (
                ("member_id", "dcpp_init_year", "time", "lat", "lon"),
                np.arange(4, dtype="float32").reshape(1, 1, 2, 1, 2) + 280,
                {"standard_name": "air_temperature", "units": "K"},
            ),
        },
        coords={
            "member_id": ["r1i1p1f1"],
            "dcpp_init_year": [0],
            "time": (
                "time",
                [0, 31],
                {"units": "days since 2000-01-01", "calendar": "standard"},
            ),
            "lat": ("lat", [10.0], {"units": "degrees_north"}),
            "lon": ("lon", [0.0, 90.0], {"units": "degrees_east"}),
            "height": (
                (),
                2.0,
                {"standard_name": "height", "units": "m", "positive": "up"},
            ),
        },
        attrs={
            "branch_time_in_child": 0.0,
            "branch_time_in_parent": 36500.0,
            "parent_time_units": "days since 0001-1-1",
        },
    ).to_netcdf(data_path)
    newer_data_path = tmp_path / "tas_v2.nc"
    shutil.copyfile(data_path, newer_data_path)

    columns = [
        "activity_id",
        "institution_id",
        "source_id",
        "experiment_id",
        "member_id",
        "table_id",
        "variable_id",
        "grid_label",
        "version",
        "path",
    ]
    with (tmp_path / "catalog.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerow(
            dict(
                zip(
                    columns,
                    [
                        "CMIP",
                        "TEST",
                        "TEST-MODEL",
                        "historical",
                        "r1i1p1f1",
                        "Amon",
                        "tas",
                        "gn",
                        "v1",
                        str(data_path),
                    ],
                    strict=True,
                ),
            ),
        )
        writer.writerow(
            dict(
                zip(
                    columns,
                    [
                        "CMIP",
                        "TEST",
                        "TEST-MODEL",
                        "historical",
                        "r1i1p1f1",
                        "Amon",
                        "tas",
                        "gn",
                        "v2",
                        str(newer_data_path),
                    ],
                    strict=True,
                ),
            ),
        )

    catalog = {
        "esmcat_version": "0.1.0",
        "id": "small-local-catalog",
        "description": "Small catalog for an end-to-end data source test",
        "catalog_file": "catalog.csv",
        "attributes": [
            {"column_name": column, "vocabulary": ""}
            for column in columns
            if column != "path"
        ],
        "assets": {"column_name": "path", "format": "netcdf"},
        "aggregation_control": {
            "variable_column_name": "variable_id",
            "groupby_attrs": [
                "activity_id",
                "institution_id",
                "source_id",
                "experiment_id",
                "member_id",
                "table_id",
                "grid_label",
            ],
            "aggregations": [
                {"type": "union", "attribute_name": "variable_id"},
            ],
        },
    }
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(catalog))
    return catalog_path


def test_configured_catalog_finds_and_loads_data(
    small_catalog: Path,
    session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A configured source should return an Iris cube from a catalog path."""
    session["projects"]["CMIP6"]["data"] = {
        "small-catalog": {
            "type": "esmvalcore.io.intake_esm.IntakeEsmDataSource",
            "catalog": str(small_catalog),
            "facets": {
                "dataset": "source_id",
                "ensemble": "member_id",
                "short_name": "variable_id",
                "version": "version",
            },
            "to_dask_kwargs": {"xarray_open_kwargs": {"decode_times": False}},
            "squeeze_dimensions": ["member_id", "dcpp_init_year"],
        },
    }
    source = esmvalcore.io.load_data_sources(session, "CMIP6")[0]
    assert isinstance(source, IntakeEsmDataSource)
    assert source.catalog == str(small_catalog)

    datasets = source.find_data(
        dataset="TEST-*",
        ensemble="*",
        short_name="tas",
        version="v1",
    )
    assert isinstance(source.catalog, intake.catalog.Catalog)
    assert len(datasets) == 1
    assert datasets[0].facets["ensemble"] == "r1i1p1f1"
    assert datasets[0].facets["version"] == "v1"

    cubes = datasets[0].to_iris()
    assert len(cubes) == 1
    assert cubes[0].shape == (2, 1, 2)
    assert cubes[0].units == "K"
    assert cubes[0].attributes["source_file"] == str(
        small_catalog.parent / "tas.nc",
    )
    assert datasets[0].attributes["source_file"] == str(
        small_catalog.parent / "tas.nc",
    )
    all_versions = source.find_data(short_name="tas", version="*")
    assert {dataset.facets["version"] for dataset in all_versions} == {
        "v1",
        "v2",
    }
    assert {dataset.name for dataset in all_versions} == {datasets[0].name}

    recipe_dataset = Dataset(
        project="CMIP6",
        dataset="TEST-MODEL",
        exp="historical",
        ensemble="r1i1p1f1",
        mip="Amon",
        short_name="tas",
        grid="gn",
        frequency="mon",
        timerange="200001/200002",
    )
    recipe_dataset.session = session
    recipe_dataset.find_files()
    assert len(recipe_dataset.files) == 1
    assert recipe_dataset.files[0].facets["version"] == "v2"
    scheduled_files = set()
    monkeypatch.setattr(recipe_module, "DOWNLOAD_FILES", scheduled_files)
    recipe_module._schedule_for_download([recipe_dataset])
    assert scheduled_files == set(recipe_dataset.files)
    cube = recipe_dataset.load()
    assert cube.shape == (2, 1, 2)


@pytest.mark.online
def test_gcs_example_loads_public_cmip6_data(session: Session) -> None:
    """The shipped cloud example should produce an Iris cube."""
    pytest.importorskip("gcsfs")
    config_path = (
        importlib.resources.files("esmvalcore.config")
        / "configurations"
        / "data-intake-esm-gcs.yml"
    )
    config = yaml.safe_load(config_path.read_text())
    source_config = config["projects"]["CMIP6"]["data"]["gcs-catalog"]
    source = IntakeEsmDataSource(
        name="gcs-catalog",
        project="CMIP6",
        priority=1,
        **{
            key: value for key, value in source_config.items() if key != "type"
        },
    )
    datasets = source.find_data(
        dataset="GFDL-CM4",
        exp="historical",
        ensemble="r1i1p1f1",
        short_name="tas",
        mip="Amon",
        grid="gr1",
        version="20180701",
    )
    assert len(datasets) == 1
    cubes = datasets[0].to_iris()
    assert len(cubes) == 1
    assert cubes[0].shape == (1980, 180, 288)
    assert cubes[0].units == "K"

    session["projects"]["CMIP6"]["data"] = {"gcs-catalog": source_config}
    recipe_dataset = Dataset(
        project="CMIP6",
        activity="CMIP",
        institute="NOAA-GFDL",
        dataset="GFDL-CM4",
        exp="historical",
        ensemble="r1i1p1f1",
        mip="Amon",
        short_name="tas",
        grid="gr1",
        version="20180701",
        frequency="mon",
        timerange="185001/185002",
    )
    recipe_dataset.session = session
    cube = recipe_dataset.load()
    assert cube.shape[0] == 2
