"""
Integration tests for :func:`esmvalcore.preprocessor._io._load_zarr`.

This is a dedicated test module for Zarr files IO; we have identified
a number of issues with Zarr IO so it deserves its own test module.
"""

import iris
import ncdata
import pytest
import xarray as xr


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

    zarr_xr = xr.open_dataset(
        zarr_path,
        consolidated=True,
        use_cftime=True,
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

    zarr_xr = xr.open_dataset(
        zarr_path,
        consolidated=False,
        use_cftime=True,
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

    zarr_xr = xr.open_dataset(
        zarr_path,
        consolidated=True,
        use_cftime=True,
        engine="zarr",
        backend_kwargs={},
    )

    conversion_func = ncdata.iris_xarray.cubes_from_xarray
    cubes = conversion_func(zarr_xr)

    assert isinstance(cubes, iris.cube.CubeList)
    assert len(cubes) == 24
