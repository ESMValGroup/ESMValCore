"""
Integration tests for :func:`esmvalcore.preprocessor._io._load_zarr`.

This is a dedicated test module for Zarr files IO; we have identified
a number of issues with Zarr IO so it deserves its own test module.
"""

import xarray as xr


def test_load_zarr_xarray():
    """
    Test loading a Zarr store from a https Object Store.

    This is a Zarr Group (multivariate file).
    This tests only the Xarray load, and not the iris cube
    conversion via ``ncdata``.
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
