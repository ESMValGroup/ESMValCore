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
    _to_path_dict,
)

try:
    import gcsfs  # noqa: F401

    gcfs_available = True
except ImportError:
    gcfs_available = False

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
        catalog=cat,
    )

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


def test_to_path_dict_nofiles() -> None:
    """Test for quiet flag.

    If we disable the `quiet` flag and pass a search query that returns no results, `to_path_dict`
    should warn.

    TODO: Can this code path ever be triggered in practice?
    """
    cat: esm_datastore = intake.open_esm_datastore(esm_ds_fhandle.as_posix())

    empty_cat = cat.search(variable_id="non_existent_variable")

    with pytest.warns(UserWarning, match="There are no datasets to load!"):
        ret = _to_path_dict(empty_cat, quiet=False)

    assert ret == {}


def test_search_time_facet_transformation() -> None:
    """Test for `time_separator` handling in `find_data`.

    Ensure that `find_data` correctly transforms time facet values to use the correct separator when searching the catalog.
    """
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

    results = data_source.find_data(timerange="185001/230012")
    dataset = results[0]
    assert isinstance(dataset, IntakeEsmDataset)

    with pytest.raises(ESMDataSourceError):
        dataset.to_iris()


"""
The following tests load some real data. These come from gcs. Mostly this is to ensure that we can
load stuff out of the cloud.
"""


@pytest.fixture(scope="session")
def pangeo_ds() -> esm_datastore:
    """Load the pangeo CMIP6 catalog as an intake_esm datastore.

    This is a remote resource, so we use session scope to avoid reloading it for every test.
    """
    return intake.open_esm_datastore(
        "https://storage.googleapis.com/cmip6/pangeo-cmip6.json",
    )


def test_remote_esm_dataset(pangeo_ds: esm_datastore) -> None:
    data_source = IntakeEsmDataSource(
        name="src",
        project="CMIP6",
        priority=1,
        time_separator="-",
        facets={
            "activity_id": "activity_id",
            "institution_id": "institution_id",
            "source_id": "source_id",
            "experiment_id": "experiment_id",
            "member_id": "member_id",
            "table_id": "table_id",
            "variable_id": "variable_id",
            "grid_label": "grid_label",
            "dcpp_init_year": "dcpp_init_year",
            "version": "version",
        },
        values={},
        catalog=pangeo_ds,
    )

    # Equivalent direct search on the catalog.
    # Note: This *might* change since it's a remote resource, so better to calculate, not specify, expected len
    expected_len = len(
        pangeo_ds.search(
            table_id="Amon",
            experiment_id="historical",
            institution_id="NOAA-GFDL",
        ),
    )

    results = data_source.find_data(
        institution_id="NOAA-GFDL",
        table_id="Amon",
        experiment_id="piControl",
    )

    assert len(results) == expected_len

    ds = results[0]
    assert isinstance(ds, IntakeEsmDataset)

    if gcfs_available:
        """
        CT NOTE
        -------
        This error is potentially gonna be a bit of an issue: comes from decoding time_bounds.
        How far do we want to go down the rabbit hole of fixing data issues versus saying 'Nope,
        not supported'?
        Might be better to fix it in xarray instead, I think it should be straightforward?
        ---

        I've also tried this with a bunch of other datasets & they all variously error out. Presumably
        this is due to iris's data model being stricter than xarrays?
        """
        with pytest.raises(
            ValueError,
            match=r"When encoding chunked arrays of datetime values, both the units and dtype must be prescribed or both must be unprescribed. Prescribing only one or the other is not currently supported. Got a units encoding of hours since 0151-01-16 12:00:00.000000 and a dtype encoding of None.",
        ):
            ds.to_iris(
                xarray_open_kwargs={"decode_cf": True, "decode_times": True},
            )
    else:
        # Can't match because it's not in the ESMDataSourceError
        with pytest.raises(
            ESMDataSourceError,
        ):
            ds.to_iris()


def test_remote_esm_dataset_keyerr(pangeo_ds: esm_datastore) -> None:
    data_source = IntakeEsmDataSource(
        name="src",
        project="CMIP6",
        priority=1,
        time_separator="-",
        facets={
            "activity_id": "activity_id",
            "institution_id": "institution_id",
            "source_id": "source_id",
            "experiment_id": "experiment_id",
            "member_id": "member_id",
            "table_id": "table_id",
            "variable_id": "variable_id",
            "grid_label": "grid_label",
            "dcpp_init_year": "dcpp_init_year",
            "version": "version",
        },
        values={},
        catalog=pangeo_ds,
    )

    # Equivalent direct search on the catalog.
    # Note: This *might* change since it's a remote resource, so better to calculate, not specify, expected len
    expected_len = len(
        pangeo_ds.search(
            table_id="Amon",
            experiment_id="historical",
            institution_id="MIROC",
            variable_id="tasmax",
        ),
    )

    results = data_source.find_data(
        institution_id="MIROC",
        table_id="Amon",
        experiment_id="historical",
        variable_id="tasmax",
    )

    assert len(results) == expected_len

    ds = results[0]
    assert isinstance(ds, IntakeEsmDataset)

    errtype = KeyError if gcfs_available else ESMDataSourceError

    with pytest.raises(
        errtype,
    ):
        ds.to_iris(
            xarray_open_kwargs={"decode_cf": True, "decode_times": True},
            xarray_combine_by_coords_kwargs={
                "coords": "minimal",
                "compat": "override",
            },
        )
