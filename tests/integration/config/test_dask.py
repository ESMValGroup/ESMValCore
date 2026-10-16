"""Integration tests for :mod:`esmvalcore.config._dask`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import dask.config
import xarray as xr

from esmvalcore.config._dask import get_distributed_client

if TYPE_CHECKING:
    from esmvalcore.config import Session


def _get_file_cache_maxsize() -> int:
    return xr.get_options()["file_cache_maxsize"]


def test_worker_xarray_options(session: Session) -> None:
    """Test that the xarray options are set on the Dask workers."""
    session["dask"] = {
        "use": "local_distributed",
        "profiles": {
            "local_distributed": {
                "cluster": {
                    "type": "distributed.LocalCluster",
                    "n_workers": 1,
                    "processes": False,
                    "dashboard_address": None,
                },
            },
        },
        "worker_xarray_options": {"file_cache_maxsize": 2},
    }
    original = xr.get_options()
    try:
        # Avoid leaking the scheduler address to other tests.
        with (
            dask.config.set({"scheduler_address": None}),
            get_distributed_client(session) as client,
        ):
            assert client is not None
            result = client.run(_get_file_cache_maxsize)
    finally:
        # The worker runs in this process, so restore the original options.
        xr.set_options(file_cache_maxsize=original["file_cache_maxsize"])
    assert list(result.values()) == [2]
