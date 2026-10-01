"""Unit tests for esmvalcore.io.intake_esm."""

from __future__ import annotations

import importlib.resources
from pathlib import Path
from typing import TYPE_CHECKING

import intake
import pytest
import xarray as xr
from intake_esm.source import ESMDataSourceError

import esmvalcore.io.intake_esm
from esmvalcore.io.intake_esm import (
    IntakeEsmDataset,
    IntakeEsmDataSource,
)

if TYPE_CHECKING:
    from intake_esm.core import esm_datastore
    from pytest_mock import MockerFixture

with importlib.resources.as_file(
    importlib.resources.files("tests"),
) as test_dir:
    esm_ds_fhandle = (
        Path(test_dir)
        / "sample_data"
        / "intake-esm"
        / "catalog"
        / "cmip6-netcdf.json"
    )


"""
These tests all use a local datastore, for which the data isn't available locally. This is mostly for
speed reasons. Anything like `.to_iris()` is going to raise a FileNotFoundError.
"""


def test_intake_esm_dataset_repr() -> None:
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
    cat = intake.open_esm_datastore(esm_ds_fhandle.as_posix()).search(
        source_id="BCC-CSM2-MR",
        experiment_id="abrupt-4xCO2",
        variable_id="tasmax",
        member_id="r1i1p1f1",
    )
    ds = xr.Dataset(attrs={"attr": "value"})
    mocker.patch.object(cat, "to_dask", return_value=ds)

    cube = mocker.Mock()
    cubes = [cube]
    mocker.patch.object(
        esmvalcore.io.intake_esm,
        "dataset_to_iris",
        return_value=cubes,
    )

    to_dask_kwargs = {"xarray_open_kwargs": {"decode_times": False}}
    dataset = IntakeEsmDataset(
        name="test",
        facets={},
        catalog=cat,
        to_dask_kwargs=to_dask_kwargs,
    )
    result = dataset.to_iris()
    assert result is cubes
    cat.to_dask.assert_called_once_with(**to_dask_kwargs)

    assert dataset.attributes == {
        "attr": "value",
        "source_file": cat.df[cat.esmcat.assets.column_name].iloc[0],
    }


def test_to_iris_rejects_multiple_members(mocker: MockerFixture) -> None:
    """Squeezing a dimension must not silently discard ensemble members."""
    cat = intake.open_esm_datastore(esm_ds_fhandle.as_posix()).search(
        source_id="BCC-CSM2-MR",
        experiment_id="abrupt-4xCO2",
        variable_id="tasmax",
        member_id="r1i1p1f1",
    )
    mocker.patch.object(
        cat,
        "to_dask",
        return_value=xr.Dataset(coords={"member_id": ["r1", "r2"]}),
    )
    dataset = IntakeEsmDataset(
        name="test",
        facets={},
        catalog=cat,
        squeeze_dimensions=("member_id",),
    )
    with pytest.raises(
        ValueError,
        match="dimension 'member_id' with length 2",
    ):
        dataset.to_iris()


def test_find_data_no_results_sets_debug_info() -> None:
    """When catalog.search returns empty results, find_data should return empty list and set debug_info."""
    cat: esm_datastore = intake.open_esm_datastore(esm_ds_fhandle.as_posix())
    data_source = IntakeEsmDataSource(
        name="src",
        project="CMIP6",
        priority=1,
        facets={"short_name": "variable_id"},
        catalog=cat,
    )

    result = data_source.find_data(short_name="non_existent_variable")
    assert result == []
    expected_debug_info = "`intake_esm.esm_datastore().search(variable_id=['non_existent_variable'])` did not return any results."
    assert data_source.debug_info == expected_debug_info


def test_find_data() -> None:
    """find_data should convert catalog.df rows into IntakeEsmDataset instances.

    CT Note: I'm not sure what project should be in here?
    """
    cat: esm_datastore = intake.open_esm_datastore(esm_ds_fhandle.as_posix())

    data_source = IntakeEsmDataSource(
        name="src",
        project="CMIP6",
        priority=1,
        facets={
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
        values={},
        to_dask_kwargs={"xarray_open_kwargs": {"decode_times": False}},
        catalog=cat,
    )

    # Two intake-esm keys contain multiple ensembles, so there are 10 elements.
    results = data_source.find_data(short_name="tasmax")
    assert isinstance(results, list)
    assert len(results) == 10
    assert len({dataset.name for dataset in results}) == 10

    dataset = next(
        dataset
        for dataset in results
        if dataset.facets["dataset"] == "BCC-CSM2-MR"
    )
    assert isinstance(dataset, IntakeEsmDataset)
    assert dataset.to_dask_kwargs == {
        "xarray_open_kwargs": {"decode_times": False},
    }
    assert dataset.name.startswith("CMIP6:")
    assert "BCC-CSM2-MR" in dataset.name
    assert len(dataset.catalog) == 1

    assert dataset.facets == {
        "activity": "CMIP",
        "dataset": "BCC-CSM2-MR",
        "ensemble": "r1i1p1f1",
        "exp": "abrupt-4xCO2",
        "grid": "gn",
        "institute": "BCC",
        "mip": "Amon",
        "short_name": "tasmax",
        "version": "v20181016",
    }


def test_to_iris_nomock():
    """`to_iris` should load data from a real intake-esm catalog."""
    cat: esm_datastore = intake.open_esm_datastore(esm_ds_fhandle.as_posix())

    data_source = IntakeEsmDataSource(
        name="src",
        project="CMIP6",
        priority=1,
        facets={
            "activity": "activity_id",
            "dataset": "source_id",
            "ensemble": "member_id",
            "exp": "experiment_id",
            "institute": "institution_id",
            "grid": "grid_label",
            "mip": "table_id",
            "short_name": "variable_id",
            "timerange": "time_range",
            "version": "version",
        },
        values={},
        catalog=cat,
    )

    # Call find_data - it should use the df we set and return 8 datasets.
    # Then we'll load the first one.
    results = data_source.find_data(short_name="tasmax")
    dataset = results[0]
    assert isinstance(dataset, IntakeEsmDataset)

    with pytest.raises(ESMDataSourceError):
        dataset.to_iris()


def test_search_time_overlap() -> None:
    """A short recipe period should match a long catalog asset period."""
    cat: esm_datastore = intake.open_esm_datastore(esm_ds_fhandle.as_posix())

    data_source = IntakeEsmDataSource(
        name="src",
        project="CMIP6",
        priority=1,
        time_separator="-",
        facets={
            "activity": "activity_id",
            "dataset": "source_id",
            "ensemble": "member_id",
            "exp": "experiment_id",
            "institute": "institution_id",
            "grid": "grid_label",
            "mip": "table_id",
            "short_name": "variable_id",
            "timerange": "time_range",
            "version": "version",
        },
        values={},
        catalog=cat,
    )

    results = data_source.find_data(
        dataset="BCC-ESM1",
        short_name="tasmax",
        timerange="200001/200002",
    )
    assert len(results) == 1
    dataset = results[0]
    assert isinstance(dataset, IntakeEsmDataset)
    assert len(dataset.catalog.df) == 1
    assert (
        data_source.find_data(
            dataset="BCC-ESM1",
            short_name="tasmax",
            timerange="240001/240002",
        )
        == []
    )

    with pytest.raises(ESMDataSourceError):
        dataset.to_iris()
