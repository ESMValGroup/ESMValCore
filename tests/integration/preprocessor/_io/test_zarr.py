"""
Integration tests for :func:`esmvalcore.preprocessor._io._load_zarr`.

This is a dedicated test module for Zarr files IO; we have identified
a number of issues with Zarr IO so it deserves its own test module.

We have a permanent bucket: esmvaltool-zarr at CEDA's object store
"url": "https://uor-aces-o.s3-ext.jc.rl.ac.uk/esmvaltool-zarr",
where will host a number of test files. Bucket is anon/anon
(read/GET-only, but PUT can be allowed). Bucket operations are done
via usual MinIO client (mc command) e.g. ``mc list``, ``mc du`` etc.

Further performance investigations are being run with a number of tests
that look at ncdata at https://github.com/valeriupredoi/esmvaltool_zarr_tests
also see https://github.com/pp-mo/ncdata/issues/139
"""

from importlib.resources import files as importlib_files
from pathlib import Path

import cf_units
import pytest

from esmvalcore.preprocessor._io import load


@pytest.mark.parametrize("input_type", [str, Path])
def test_load_zarr2_local(input_type):
    """Test loading a Zarr2 store from local FS."""
    zarr_path = (
        Path(importlib_files("tests"))
        / "sample_data"
        / "zarr-sample-data"
        / "example_field_0.zarr2"
    )

    cubes = load(input_type(zarr_path))

    assert len(cubes) == 1
    cube = cubes[0]
    assert cube.var_name == "q"
    assert cube.standard_name == "specific_humidity"
    assert cube.long_name is None
    assert cube.units == cf_units.Unit("1")
    coords = cube.coords()
    coord_names = [coord.standard_name for coord in coords]
    assert "longitude" in coord_names
    assert "latitude" in coord_names


def test_load_zarr2_remote():
    """Test loading a Zarr2 store from a https Object Store."""
    zarr_path = (
        "https://uor-aces-o.s3-ext.jc.rl.ac.uk/"
        "esmvaltool-zarr/example_field_0.zarr2"
    )

    # with "dummy" storage options
    cubes = load(
        zarr_path,
        ignore_warnings=None,
        backend_kwargs={"storage_options": {}},
    )

    assert len(cubes) == 1
    cube = cubes[0]
    assert cube.var_name == "q"
    assert cube.standard_name == "specific_humidity"
    assert cube.long_name is None
    assert cube.units == cf_units.Unit("1")
    coords = cube.coords()
    coord_names = [coord.standard_name for coord in coords]
    assert "longitude" in coord_names
    assert "latitude" in coord_names

    # without storage_options
    cubes = load(zarr_path)

    assert len(cubes) == 1
    cube = cubes[0]
    assert cube.var_name == "q"
    assert cube.standard_name == "specific_humidity"
    assert cube.long_name is None
    assert cube.units == cf_units.Unit("1")
    coords = cube.coords()
    coord_names = [coord.standard_name for coord in coords]
    assert "longitude" in coord_names
    assert "latitude" in coord_names


def test_load_zarr3_remote():
    """Test loading a Zarr3 store from a https Object Store."""
    zarr_path = (
        "https://uor-aces-o.s3-ext.jc.rl.ac.uk/"
        "esmvaltool-zarr/example_field_0.zarr3"
    )

    # with "dummy" storage options
    cubes = load(
        zarr_path,
        ignore_warnings=None,
        backend_kwargs={"storage_options": {}},
    )

    assert len(cubes) == 1
    cube = cubes[0]
    assert cube.var_name == "q"
    assert cube.standard_name == "specific_humidity"
    assert cube.long_name is None
    assert cube.units == cf_units.Unit("1")
    coords = cube.coords()
    coord_names = [coord.standard_name for coord in coords]
    assert "longitude" in coord_names
    assert "latitude" in coord_names


def test_load_zarr3_cmip6_metadata():
    """
    Test loading a Zarr3 store from a https Object Store.

    This test loads just the metadata, no computations.

    This is an actual CMIP6 dataset (Zarr built from netCDF4 via Xarray)
    - Zarr store on disk: 243 MiB
    - compression: Blosc
    - Dimensions: (lat: 128, lon: 256, time: 2352, axis_nbounds: 2)
    - chunking: time-slices; netCDF4.Dataset.chunking() = [1, 128, 256]

    Test takes 8-9s (median: 8.5s) and needs max Res mem: 1GB
    """
    zarr_path = (
        "https://uor-aces-o.s3-ext.jc.rl.ac.uk/"
        "esmvaltool-zarr/pr_Amon_CNRM-ESM2-1_02Kpd-11_r1i1p2f2_gr_200601-220112.zarr3"
    )

    # with "dummy" storage options
    cubes = load(
        zarr_path,
        ignore_warnings=None,
        backend_kwargs={"storage_options": {}},
    )

    assert len(cubes) == 1
    cube = cubes[0]
    assert cube.var_name == "pr"
    assert cube.standard_name == "precipitation_flux"
    assert cube.long_name == "Precipitation"
    assert cube.units == cf_units.Unit("kg m-2 s-1")
    assert cube.has_lazy_data()


def test_load_zarr_remote_not_zarr_file():
    """
    Test loading a Zarr store from a https Object Store.

    This fails due to the file being loaded is not a Zarr file.
    """
    zarr_path = (
        "https://uor-aces-o.s3-ext.jc.rl.ac.uk/"
        "esmvaltool-zarr/example_field_0.zarr17"
    )

    msg = (
        "File 'https://uor-aces-o.s3-ext.jc.rl.ac.uk/"
        "esmvaltool-zarr/example_field_0.zarr17' can not "
        "be opened as Zarr file at the moment."
    )
    with pytest.raises(ValueError, match=msg):
        load(zarr_path)


def test_load_zarr_remote_not_file():
    """
    Test loading a Zarr store from a https Object Store.

    This fails due to non-existing file.
    """
    zarr_path = (
        "https://uor-aces-o.s3-ext.jc.rl.ac.uk/"
        "esmvaltool-zarr/example_field_0.zarr22"
    )

    msg = (
        "File 'https://uor-aces-o.s3-ext.jc.rl.ac.uk/"
        "esmvaltool-zarr/example_field_0.zarr22' can not "
        "be opened as Zarr file at the moment."
    )
    with pytest.raises(ValueError, match=msg):
        load(zarr_path)


def test_load_zarr_local_not_file():
    """
    Test loading something that has a zarr extension.

    But file doesn't exist (on local FS).
    """
    zarr_path = "esmvaltool-zarr/example_field_0.zarr22"

    # "Unable to find group" or "No group found"
    # Zarr keeps changing the exception string so matching
    # is bound to fail the test
    with pytest.raises(FileNotFoundError):
        load(zarr_path)
