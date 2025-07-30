"""
Integration tests for :func:`esmvalcore.preprocessor._io._load_zarr`.

This is a dedicated test module for Zarr files IO; we have identified
a number of issues with Zarr IO so it deserves its own test module.
"""

import cf_units
import iris
import ncdata
import pytest
import xarray as xr

from esmvalcore.preprocessor._io import load


def test_load_zarr_xarray():
    """
    Test loading a Zarr store from a https Object Store.

    This tests only the Xarray load, and not the iris cube
    conversion via ``ncdata``.

    This is a Zarr Group (multivariate file: 24 variables).
    It is on a HEALPIX grid.

    $ mc du bryan/esmvaltool-zarr/um.PT1H.hp_z2.zarr
    125MiB	132 objects	esmvaltool-zarr/um.PT1H.hp_z2.zarr
    Loading in Xarry takes 1.5-2s and ingests a max RES mem 430MB.
    No issues on CircleCI (pytest -n 4).
    """
    zarr_path = (
        "https://uor-aces-o.s3-ext.jc.rl.ac.uk/"
        "esmvaltool-zarr/um.PT1H.hp_z2.zarr"
    )

    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    zarr_xr = xr.open_dataset(
        zarr_path,
        consolidated=True,
        decode_times=time_coder,
        engine="zarr",
        backend_kwargs={},
    )

    assert zarr_xr["tas"].any()


def test_load_zarr_to_iris_via_ncdata_consolidated_false():
    """
    Test loading a Zarr store from a https Object Store.

    Same test as ``test_load_zarr_xarray`` only this time we are
    passing the Xarray Dataset to ``ncdata``.

    Test needs about 700MB max RES memory, takes 6.5-7s.
    Hangs (4 in 6) on CircleCI (pytest -n 4),
    see https://github.com/ESMValGroup/ESMValCore/pull/2785/
    checks?check_run_id=46944622218

    Hangs in single proc on CircleCI (pytest simple).

    The ONLY way the test doesn't hang on CircleCI is to set
    consolidate=False (or, not use it as kwarg at all). But that
    returns No Cubes!

    See https://github.com/pp-mo/ncdata/issues/138
    """
    zarr_path = (
        "https://uor-aces-o.s3-ext.jc.rl.ac.uk/"
        "esmvaltool-zarr/um.PT1H.hp_z2.zarr"
    )

    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    zarr_xr = xr.open_dataset(
        zarr_path,
        consolidated=False,
        decode_times=time_coder,
        engine="zarr",
        backend_kwargs={},
    )

    conversion_func = ncdata.iris_xarray.cubes_from_xarray
    cubes = conversion_func(zarr_xr)

    assert isinstance(cubes, iris.cube.CubeList)
    assert not cubes


@pytest.mark.skip(reason="Hangs on CircleCI, see test description.")
def test_load_zarr_to_iris_via_ncdata_consolidated_true():
    """
    Test loading a Zarr store from a https Object Store.

    Same test as ``test_load_zarr_xarray`` only this time we are
    passing the Xarray Dataset to ``ncdata``.

    Test needs about 700MB max RES memory, takes 6.5-7s.
    Hangs (4 in 6) on CircleCI (pytest -n 4),
    see https://github.com/ESMValGroup/ESMValCore/pull/2785/
    checks?check_run_id=46944622218

    Hangs (4 in 6) in single proc on CircleCI (pytest simple).

    The ONLY way the test doesn't hang on CircleCI is to set
    consolidate=False (or, not use it as kwarg at all). But that
    returns No Cubes!

    See https://github.com/pp-mo/ncdata/issues/138
    """
    zarr_path = (
        "https://uor-aces-o.s3-ext.jc.rl.ac.uk/"
        "esmvaltool-zarr/um.PT1H.hp_z2.zarr"
    )

    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    zarr_xr = xr.open_dataset(
        zarr_path,
        consolidated=True,
        decode_times=time_coder,
        engine="zarr",
        backend_kwargs={},
    )

    conversion_func = ncdata.iris_xarray.cubes_from_xarray
    cubes = conversion_func(zarr_xr)

    assert isinstance(cubes, iris.cube.CubeList)
    assert len(cubes) == 24


def test_load_zarr3():
    """
    Test loading a Zarr3 store from a https Object Store.

    We have a permanent bucket: esmvaltool-zarr at CEDA's object store
    "url": "https://uor-aces-o.s3-ext.jc.rl.ac.uk",
    where will host a number of test files like this one.
    """
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

    We have a permanent bucket: esmvaltool-zarr at CEDA's object store
    "url": "https://uor-aces-o.s3-ext.jc.rl.ac.uk",
    where will host a number of test files like this one.

    This is an actual CMIP6 dataset (Zarr built from netCDF4 via Xarray)
    - Zarr store on disk: 243 MiB
    - compression: Blosc
    - Dimensions: (lat: 128, lon: 256, time: 2352, axis_nbounds: 2)
    - chunking: time-slices; netCDF4.Dataset.chunking() = [1, 128, 256]
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
