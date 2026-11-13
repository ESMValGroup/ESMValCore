"""Tests for :mod:`esmvalcore.io.xarray`."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import cftime
import iris.cube
import numpy as np
import pytest
import xarray as xr

from esmvalcore.io.xarray import (
    XarrayDataset,
    XarrayDataSource,
    _date_bounds,
    _expand_env_vars,
    _format_template,
    _parse_template,
    _template_to_regex,
)

if TYPE_CHECKING:
    from pathlib import Path


def create_store(
    path: Path,
    variables: tuple[str, ...] = ("tas", "pr"),
    years: tuple[int, int] = (2000, 2002),
    experiments: list[str] | None = None,
    attrs: dict | None = None,
) -> str:
    """Create a small zarr store with monthly data."""
    n_months = 12 * (years[1] - years[0] + 1)
    time = np.arange(n_months) * 30.0 + 15.0
    time_bnds = np.stack([time - 15.0, time + 15.0], axis=-1)
    lat = np.array([-45.0, 45.0])
    lon = np.array([90.0, 270.0])
    dims: tuple[str, ...] = ("time", "lat", "lon")
    shape: tuple[int, ...] = (n_months, 2, 2)
    coords: dict = {}
    if experiments is not None:
        dims = ("experiment_id", *dims)
        shape = (len(experiments), *shape)
        coords["experiment_id"] = ("experiment_id", experiments)
    data_vars: dict[str, tuple] = {
        name: (
            dims,
            np.full(shape, i, dtype=np.float32),
            {"units": "1", "cell_measures": "area: areacella"},
        )
        for i, name in enumerate(variables)
    }
    data_vars["areacella"] = (("lat", "lon"), np.ones((2, 2)), {"units": "m2"})
    data_vars["time_bnds"] = (("time", "bnds"), time_bnds)
    data_vars["lat_bnds"] = (("lat", "bnds"), [[-90.0, 0.0], [0.0, 90.0]])
    dataset = xr.Dataset(
        data_vars,
        coords={
            **coords,
            "time": (
                "time",
                time,
                {
                    "units": f"days since {years[0]}-01-01",
                    "calendar": "360_day",
                    "standard_name": "time",
                    "bounds": "time_bnds",
                },
            ),
            "lat": (
                "lat",
                lat,
                {
                    "units": "degrees_north",
                    "standard_name": "latitude",
                    "bounds": "lat_bnds",
                },
            ),
            "lon": (
                "lon",
                lon,
                {"units": "degrees_east", "standard_name": "longitude"},
            ),
        },
        attrs=attrs or {},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_zarr(path, consolidated=True, zarr_format=2)
    return str(path)


OPTIONS = {"engine": "zarr", "chunks": {}}


def make_source(**kwargs: Any) -> XarrayDataSource:
    return XarrayDataSource(
        name="test",
        project="test",
        priority=1,
        **({"options": OPTIONS} | kwargs),
    )


def test_format_template():
    result = _format_template(
        "{project}/{dataset}/{exp.upper}.zarr",
        {"project": "CMIP6", "exp": ["a", "b"]},
        missing="*",
    )
    assert result == ["CMIP6/*/A.zarr", "CMIP6/*/B.zarr"]


def test_format_template_missing_facet():
    with pytest.raises(KeyError, match="'dataset' has not been specified"):
        _format_template("{dataset}", {})


@pytest.mark.parametrize(
    ("template", "text", "expected"),
    [
        (
            "s3://bucket/{dataset}_{exp}.zarr",
            "s3://bucket/ESACCI-L4_GHRSST_x.zarr",
            {"dataset": "ESACCI-L4", "exp": "GHRSST_x"},
        ),
        (
            "{project}.{activity}.{short_name}.{version}",
            "CMIP6.CMIP.tas.v20200616",
            {
                "project": "CMIP6",
                "activity": "CMIP",
                "short_name": "tas",
                "version": "v20200616",
            },
        ),
        ("{a}/{a}.zarr", "x/x.zarr", {"a": "x"}),
        ("{a}/{a}.zarr", "x/y.zarr", None),
        ("https://*/{a}.zarr", "https://host/x.zarr", {"a": "x"}),
        ("{a}.zarr", "x.zarr/extra", None),
    ],
)
def test_parse_template(template, text, expected):
    assert _parse_template(_template_to_regex(template), text) == expected


def test_expand_env_vars(monkeypatch):
    monkeypatch.setenv("MY_TOKEN", "secret")
    monkeypatch.delenv("UNSET_VARIABLE", raising=False)
    options = {
        "storage_options": {
            "headers": {"Authorization": "Bearer ${MY_TOKEN}"},
            "other": ["${UNSET_VARIABLE}", 1],
        },
    }
    assert _expand_env_vars(options) == {
        "storage_options": {
            "headers": {"Authorization": "Bearer secret"},
            "other": ["${UNSET_VARIABLE}", 1],
        },
    }


def test_env_vars_expanded_in_options(monkeypatch):
    monkeypatch.setenv("MY_TOKEN", "secret")
    source = make_source(
        store="x.zarr",
        options={"storage_options": {"token": "${MY_TOKEN}"}},
    )
    assert source.options == {"storage_options": {"token": "secret"}}


@pytest.mark.parametrize(
    ("timerange", "start", "end"),
    [
        ("2000/2001", (2000, 1, 1), (2002, 1, 1)),
        ("200011/200012", (2000, 11, 1), (2001, 1, 1)),
        ("20000101/20001230", (2000, 1, 1), (2001, 1, 1)),
        (
            "20000101T00/20000101T23",
            (2000, 1, 1),
            (2000, 1, 2),
        ),
        # 2000-01-01 plus one year, i.e. up to and including 2001-01-01.
        ("2000/P1Y", (2000, 1, 1), (2001, 1, 2)),
    ],
)
def test_date_bounds(timerange, start, end):
    start_date, end_date = _date_bounds(timerange, "360_day")
    assert start_date == cftime.datetime(*start, calendar="360_day")
    assert end_date == cftime.datetime(*end, calendar="360_day")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"store": "a", "stores": ["b"]},
        {"stores": ["b"], "index": "c"},
        {},
        {"store": "a", "facets_from_attribute": {"attribute": "x"}},
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError, match="test"):
        make_source(**kwargs)


def test_find_data_single_store(tmp_path):
    store = create_store(tmp_path / "data.zarr", variables=("t2m", "tp", "x"))
    source = make_source(
        store=store,
        variables={"tas": "t2m", "pr": ["tp"], "missing": ["tp", "absent"]},
        facets={"dataset": "ERA5"},
    )
    result = source.find_data(short_name="*", dataset="ERA5")
    # Bounds are not offered as variables, cell measures are.
    assert sorted(r.facets["short_name"] for r in result) == [
        "areacella",
        "pr",
        "tas",
        "x",
    ]
    tas = next(r for r in result if r.facets["short_name"] == "tas")
    assert tas.name == f"{store}/tas"
    assert tas.variables == ["t2m"]
    assert tas.facets == {
        "dataset": "ERA5",
        "short_name": "tas",
        "timerange": "20000116T000000/20021216T000000",
    }


def test_find_data_no_match(tmp_path):
    store = create_store(tmp_path / "data.zarr")
    source = make_source(store=store, facets={"dataset": "ERA5"})
    assert source.find_data(short_name="tas", dataset="other") == []
    assert source.find_data(short_name="absent") == []
    assert "No data matching" in source.debug_info


def test_find_data_url_template_glob(tmp_path):
    create_store(tmp_path / "A" / "historical.zarr")
    create_store(tmp_path / "A" / "ssp585.zarr", years=(2015, 2016))
    create_store(tmp_path / "B" / "historical.zarr")
    source = make_source(url_template=f"{tmp_path}/{{dataset}}/{{exp}}.zarr")

    result = source.find_data(short_name="tas", dataset="A", exp="*")
    assert [(r.facets["dataset"], r.facets["exp"]) for r in result] == [
        ("A", "historical"),
        ("A", "ssp585"),
    ]

    result = source.find_data(short_name="tas", exp="historical")
    assert [r.facets["dataset"] for r in result] == ["A", "B"]


def test_find_data_stores_and_timerange(tmp_path):
    stores = [
        create_store(tmp_path / "historical.zarr", years=(2000, 2002)),
        create_store(tmp_path / "future.zarr", years=(2015, 2016)),
    ]
    source = make_source(
        stores=stores,
        url_template=f"{tmp_path}/{{exp}}.zarr",
    )
    result = source.find_data(short_name="tas", timerange="2001/2003")
    assert [r.facets["exp"] for r in result] == ["historical"]
    assert result[0].timerange == "2001/2003"


def test_find_data_index(tmp_path):
    create_store(tmp_path / "bucket" / "A.zarr")
    create_store(tmp_path / "bucket" / "B.zarr")
    index = tmp_path / "bucket" / "data_ids.json"
    index.write_text(json.dumps(["A.zarr", "B.zarr"]), encoding="utf-8")
    source = make_source(
        index=str(index),
        url_template=f"{tmp_path}/bucket/{{dataset_id}}.zarr",
    )
    result = source.find_data(short_name="pr", dataset_id="B")
    assert len(result) == 1
    assert result[0].store == f"{tmp_path}/bucket/B.zarr"


def test_find_data_facets_from_attribute_and_select(tmp_path):
    ids = [
        f"CMIP6.ScenarioMIP.MOD.{exp}.{var}.v{version}"
        for exp, version in [("ssp126", 1), ("ssp585", 2)]
        for var in ["tas", "pr"]
    ]
    store = create_store(
        tmp_path / "data.zarr",
        experiments=["ssp126", "ssp585"],
        attrs={"dataset_ids": ids},
    )
    source = make_source(
        store=store,
        facets_from_attribute={
            "attribute": "dataset_ids",
            "template": "{project}.{activity}.{dataset}.{exp}.{short_name}.{version}",
        },
        select={"exp": "experiment_id"},
        name_template="{project}.{activity}.{dataset}.{exp}.{short_name}",
    )
    result = source.find_data(short_name="tas", exp="ssp585")
    assert len(result) == 1
    element = result[0]
    assert element.name == "CMIP6.ScenarioMIP.MOD.ssp585.tas"
    assert element.facets["version"] == "v2"
    assert element.select == {"experiment_id": "ssp585"}


def test_find_data_select_without_attribute(tmp_path):
    store = create_store(tmp_path / "data.zarr", experiments=["a", "b"])
    source = make_source(store=store, select={"exp": "experiment_id"})
    result = source.find_data(short_name="tas")
    assert [r.name for r in result] == [f"{store}/tas/a", f"{store}/tas/b"]


def test_to_iris(tmp_path):
    store = create_store(
        tmp_path / "data.zarr",
        variables=("t2m", "tp"),
        experiments=["a", "b"],
    )
    source = make_source(
        store=store,
        variables={"tas": "t2m"},
        select={"exp": "experiment_id"},
    )
    (element,) = source.find_data(
        short_name="tas",
        exp="b",
        timerange="2001/2001",
    )
    cubes = element.to_iris()
    assert isinstance(cubes, iris.cube.CubeList)
    assert len(cubes) == 1
    cube = cubes[0]
    assert cube.var_name == "t2m"
    assert cube.has_lazy_data()
    assert cube.shape == (12, 2, 2)
    # Values of t2m are 0, for the variable that was not requested 1.
    assert cube.data.max() == 0
    time = cube.coord("time")
    assert time.has_bounds()
    assert time.cell(0).point == cftime.datetime(
        2001,
        1,
        16,
        calendar="360_day",
    )
    assert cube.coord("latitude").has_bounds()
    assert [m.var_name for m in cube.cell_measures()] == ["areacella"]
    assert not cube.coords("experiment_id")
    assert element.attributes["source_file"] == store


def test_find_data_drop_variables(tmp_path):
    store = create_store(tmp_path / "data.zarr", variables=("t2m", "tp"))
    source = make_source(store=store, variables={"tas": "t2m"})
    (element,) = source.find_data(short_name="tas")
    # Variables that t2m refers to through CF attributes are kept.
    assert element.drop_variables == ["tp"]


@pytest.mark.parametrize(
    ("options", "chunks"),
    [
        ({}, "auto"),
        ({"chunks": {}}, {}),
    ],
)
def test_to_iris_open_options(tmp_path, mocker, options, chunks):
    store = create_store(tmp_path / "data.zarr", variables=("t2m", "tp"))
    source = make_source(
        store=store,
        variables={"tas": "t2m"},
        options={"engine": "zarr"} | options,
    )
    (element,) = source.find_data(short_name="tas")
    open_dataset = mocker.spy(xr, "open_dataset")
    element.to_iris()
    open_dataset.assert_called_once_with(
        store,
        engine="zarr",
        chunks=chunks,
        decode_times=False,
        drop_variables=["tp"],
    )


def test_attributes_before_to_iris():
    element = XarrayDataset(name="x", facets={}, store="x.zarr")
    with pytest.raises(ValueError, match="Attributes have not been read yet"):
        _ = element.attributes


def test_hash():
    element = XarrayDataset(name="x", facets={"version": "v1"}, store="x")
    assert hash(element) == hash(("x", "v1"))
