"""Unit tests for esmvalcore.io.intake_esm."""

from __future__ import annotations

import importlib.resources
from pathlib import Path
from typing import TYPE_CHECKING

import intake
import pytest
import xarray as xr

import esmvalcore.io.intake_esm
from esmvalcore.io.intake_esm import IntakeEsmDataset, IntakeEsmDataSource

if TYPE_CHECKING:
    from intake_esm.core import esm_datastore
    from pytest_mock import MockerFixture


with importlib.resources.as_file(
    importlib.resources.files("tests"),
) as test_dir:
    esm_ds_fhandle = (
        Path(test_dir) / "sample_data" / "intake-esm" / "catalog" / "cmip6-netcdf.json"
    )


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


def test_find_data_no_results_sets_debug_info() -> None:
    """When catalog.search returns empty results, find_data should return empty list and set debug_info."""
    data_source = IntakeEsmDataSource(
        name="src",
        project="CMIP6",
        priority=1,
        facets={"short_name": "variable_id"},
    )

    cat: esm_datastore = intake.open_esm_datastore(esm_ds_fhandle.as_posix())

    data_source.catalog = cat

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
    )
    data_source.catalog = cat

    # Call find_data - it should use the df we set and return 8 datasets
    results = data_source.find_data(short_name="tasmax")
    assert isinstance(results, list)
    assert len(results) == 8

    dataset = results[0]
    assert isinstance(dataset, IntakeEsmDataset)
    assert dataset.name == "CMIP.BCC.BCC-CSM2-MR.abrupt-4xCO2.Amon.gn"

    assert hash(dataset) == hash((dataset.name, "v20181016"))

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


@pytest.mark.online
# @pytest.mark.skip(reason="Requires intake-esm catalog configuration")
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
            "version": "version",
        },
        values={},
    )
    data_source.catalog = cat

    # Call find_data - it should use the df we set and return 8 datasets.
    # Then we'll load the first one.
    results = data_source.find_data(short_name="tasmax")
    dataset = results[0]
    assert isinstance(dataset, IntakeEsmDataset)

    # Raises a KeyError because the dtype of the dataset is Object, which I don't think NCData likes.
    with pytest.raises(KeyError, match="'O'"):
        dataset.to_iris()
