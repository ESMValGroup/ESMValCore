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


@pytest.fixture
def chunked_catalog(tmp_path: Path) -> Path:
    """Create a catalog with different columns and time-chunk aggregation."""
    columns = [
        "activity",
        "center",
        "model",
        "experiment",
        "member",
        "table",
        "variable",
        "grid",
        "release",
        "period",
        "uri",
    ]
    rows = []
    for period, times in (
        ("200001-200002", [0, 31]),
        ("200101-200102", [366, 397]),
    ):
        path = tmp_path / f"tas_{period}.nc"
        xr.Dataset(
            {
                "tas": (
                    ("time", "lat", "lon"),
                    np.arange(4, dtype="float32").reshape(2, 1, 2) + 280,
                    {"standard_name": "air_temperature", "units": "K"},
                ),
            },
            coords={
                "time": (
                    "time",
                    times,
                    {"units": "days since 2000-01-01", "calendar": "standard"},
                ),
                "lat": ("lat", [10.0], {"units": "degrees_north"}),
                "lon": ("lon", [0.0, 90.0], {"units": "degrees_east"}),
                "height": (
                    (),
                    2.0,
                    {
                        "standard_name": "height",
                        "units": "m",
                        "positive": "up",
                    },
                ),
            },
        ).to_netcdf(path)
        row = {
            "activity": "CMIP",
            "center": "TEST",
            "model": "TEST-MODEL",
            "experiment": "historical",
            "member": "r1i1p1f1",
            "table": "Amon",
            "variable": "tas",
            "grid": "gn",
            "release": "v1",
            "period": period,
            "uri": str(path),
        }
        rows.append(row)

    with (tmp_path / "chunked.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    descriptor = {
        "esmcat_version": "0.1.0",
        "id": "chunked-local-catalog",
        "description": "Two time chunks with an alternative metadata schema",
        "catalog_file": "chunked.csv",
        "attributes": [
            {"column_name": column, "vocabulary": ""}
            for column in columns
            if column != "uri"
        ],
        "assets": {"column_name": "uri", "format": "netcdf"},
        "aggregation_control": {
            "variable_column_name": "variable",
            "groupby_attrs": [
                "activity",
                "center",
                "model",
                "experiment",
                "table",
                "grid",
            ],
            "aggregations": [
                {"type": "join_new", "attribute_name": "member"},
                {
                    "type": "join_existing",
                    "attribute_name": "period",
                    "options": {"dim": "time"},
                },
                {"type": "union", "attribute_name": "variable"},
            ],
        },
    }
    path = tmp_path / "chunked.json"
    path.write_text(json.dumps(descriptor))
    return path


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
                "activity": "activity_id",
                "dataset": "source_id",
                "ensemble": "member_id",
                "exp": "experiment_id",
                "institute": "institution_id",
                "grid": "grid_label",
                "mip": "table_id",
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


def test_chunked_catalog_with_alternative_schema(
    chunked_catalog: Path,
    small_catalog: Path,
    session: Session,
) -> None:
    """Search, aggregation, and identity should work across catalog schemas."""
    source = IntakeEsmDataSource(
        name="chunked",
        project="CMIP6",
        priority=1,
        catalog=chunked_catalog,
        facets={
            "activity": "activity",
            "institute": "center",
            "dataset": "model",
            "exp": "experiment",
            "ensemble": "member",
            "mip": "table",
            "short_name": "variable",
            "grid": "grid",
            "version": "release",
            "timerange": "period",
        },
        to_dask_kwargs={"xarray_open_kwargs": {"decode_times": False}},
        squeeze_dimensions=("member",),
    )
    all_chunks = source.find_data(dataset="TEST-MODEL", short_name="tas")
    assert len(all_chunks) == 1
    assert len(all_chunks[0].catalog.df) == 2
    assert all_chunks[0].to_iris()[0].shape == (4, 1, 2)

    first_chunk = source.find_data(
        dataset="TEST-MODEL",
        short_name="tas",
        timerange="200001/200002",
    )
    assert len(first_chunk) == 1
    assert len(first_chunk[0].catalog.df) == 1
    assert first_chunk[0].to_iris()[0].shape == (2, 1, 2)
    assert (
        source.find_data(
            dataset="TEST-MODEL",
            short_name="tas",
            timerange="199001/199002",
        )
        == []
    )

    session["projects"]["CMIP6"]["data"] = {
        "chunked": {
            "type": "esmvalcore.io.intake_esm.IntakeEsmDataSource",
            "catalog": str(chunked_catalog),
            "facets": source.facets,
            "to_dask_kwargs": source.to_dask_kwargs,
            "squeeze_dimensions": ["member"],
        },
    }
    recipe_dataset = Dataset(
        project="CMIP6",
        dataset="TEST-MODEL",
        exp="historical",
        ensemble="r1i1p1f1",
        mip="Amon",
        short_name="tas",
        grid="gn",
        version="v1",
        frequency="mon",
        timerange="200001/200002",
    )
    recipe_dataset.session = session
    assert recipe_dataset.load().shape == (2, 1, 2)

    other = IntakeEsmDataSource(
        name="other",
        project="CMIP6",
        priority=2,
        catalog=small_catalog,
        facets={
            "activity": "activity_id",
            "institute": "institution_id",
            "dataset": "source_id",
            "exp": "experiment_id",
            "ensemble": "member_id",
            "mip": "table_id",
            "short_name": "variable_id",
            "grid": "grid_label",
            "version": "version",
        },
    )
    assert (
        first_chunk[0].name
        == other.find_data(
            dataset="TEST-MODEL",
            short_name="tas",
            version="v1",
        )[0].name
    )


def test_invalid_catalog_mapping_reports_missing_column(
    chunked_catalog: Path,
) -> None:
    """A configuration for a different catalog must fail before searching."""
    source = IntakeEsmDataSource(
        name="invalid",
        project="CMIP6",
        priority=1,
        catalog=chunked_catalog,
        facets={"dataset": "source_id"},
    )
    with pytest.raises(
        ValueError,
        match="missing configured facet columns: source_id",
    ):
        source.find_data(dataset="TEST-MODEL")


def test_numeric_catalog_facet_matches_recipe_string(
    chunked_catalog: Path,
) -> None:
    """Catalog dtypes should not change which recipe values match."""
    csv_path = chunked_catalog.parent / "chunked.csv"
    with csv_path.open(newline="") as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames
        rows = list(reader)
    assert columns is not None
    for row in rows:
        row["release"] = "20200101"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    source = IntakeEsmDataSource(
        name="numeric",
        project="CMIP6",
        priority=1,
        catalog=chunked_catalog,
        facets={"dataset": "model", "version": "release"},
    )
    datasets = source.find_data(dataset="TEST-MODEL", version="20200101")
    assert len(datasets) == 1
    assert str(datasets[0].facets["version"]) == "20200101"


def test_list_valued_mapped_facet_reports_unsupported_catalog(
    tmp_path: Path,
) -> None:
    """List-valued catalog facets need a distinct selection strategy."""
    descriptor = {
        "esmcat_version": "0.1.0",
        "id": "multi-variable",
        "description": "One asset contains multiple variables",
        "catalog_dict": [
            {
                "model": "TEST-MODEL",
                "variable": ["tas", "pr"],
                "uri": str(tmp_path / "unused.nc"),
            },
        ],
        "attributes": [
            {"column_name": column, "vocabulary": ""}
            for column in ("model", "variable")
        ],
        "assets": {"column_name": "uri", "format": "netcdf"},
        "aggregation_control": {
            "variable_column_name": "variable",
            "groupby_attrs": ["model"],
            "aggregations": [
                {"type": "union", "attribute_name": "variable"},
            ],
        },
    }
    path = tmp_path / "multi.json"
    path.write_text(json.dumps(descriptor))
    source = IntakeEsmDataSource(
        name="multi",
        project="CMIP6",
        priority=1,
        catalog=path,
        facets={"dataset": "model", "short_name": "variable"},
    )
    with pytest.raises(
        ValueError,
        match="list-valued mapped facet columns: variable",
    ):
        source.find_data(dataset="TEST-MODEL", short_name="tas")


def test_ambiguous_catalog_aggregation_fails(
    chunked_catalog: Path,
) -> None:
    """Separate catalog keys must not silently discard time chunks."""
    descriptor = json.loads(chunked_catalog.read_text())
    descriptor["aggregation_control"]["groupby_attrs"].append("period")
    chunked_catalog.write_text(json.dumps(descriptor))
    source = IntakeEsmDataSource(
        name="ambiguous",
        project="CMIP6",
        priority=1,
        catalog=chunked_catalog,
        facets={
            "dataset": "model",
            "short_name": "variable",
            "timerange": "period",
        },
    )
    with pytest.raises(ValueError, match="into 2 intake-esm keys"):
        source.find_data(dataset="TEST-MODEL", short_name="tas")


def test_catalogs_deduplicate_by_logical_identity(
    chunked_catalog: Path,
    small_catalog: Path,
    session: Session,
) -> None:
    """Equivalent data in differently keyed catalogs should deduplicate."""
    session["search_data"] = "complete"
    session["projects"]["CMIP6"]["data"] = {
        "chunked": {
            "type": "esmvalcore.io.intake_esm.IntakeEsmDataSource",
            "priority": 1,
            "catalog": str(chunked_catalog),
            "facets": {
                "activity": "activity",
                "institute": "center",
                "dataset": "model",
                "exp": "experiment",
                "ensemble": "member",
                "mip": "table",
                "short_name": "variable",
                "grid": "grid",
                "version": "release",
                "timerange": "period",
            },
            "squeeze_dimensions": ["member"],
        },
        "single-file": {
            "type": "esmvalcore.io.intake_esm.IntakeEsmDataSource",
            "priority": 2,
            "catalog": str(small_catalog),
            "facets": {
                "activity": "activity_id",
                "institute": "institution_id",
                "dataset": "source_id",
                "exp": "experiment_id",
                "ensemble": "member_id",
                "mip": "table_id",
                "short_name": "variable_id",
                "grid": "grid_label",
                "version": "version",
            },
            "squeeze_dimensions": ["member_id", "dcpp_init_year"],
        },
    }
    common = {
        "project": "CMIP6",
        "dataset": "TEST-MODEL",
        "exp": "historical",
        "ensemble": "r1i1p1f1",
        "mip": "Amon",
        "short_name": "tas",
        "grid": "gn",
        "frequency": "mon",
        "timerange": "200001/200002",
    }
    selected = Dataset(**common, version="v1")
    selected.session = session
    selected.find_files()
    assert len(selected.files) == 1
    assert selected.files[0].catalog.esmcat.id == "chunked-local-catalog"

    latest = Dataset(**common)
    latest.session = session
    latest.find_files()
    assert len(latest.files) == 1
    assert latest.files[0].facets["version"] == "v2"
    assert latest.files[0].catalog.esmcat.id == "small-local-catalog"


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
