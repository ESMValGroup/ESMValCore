"""Test :mod:`esmvalcore.preprocessor._dask_progress`."""

import logging
import time

import dask
import distributed
import pytest

from esmvalcore.preprocessor import _dask_progress


@pytest.mark.parametrize("use_distributed", [False, True])
@pytest.mark.parametrize("interval", [-1, 0.0, 0.2])
def test_compute_with_progress(
    capsys,
    caplog,
    monkeypatch,
    use_distributed,
    interval,
):
    caplog.set_level(logging.INFO)
    if use_distributed:
        client = distributed.Client(n_workers=1, threads_per_worker=1)
    else:
        client = None

    monkeypatch.setitem(_dask_progress.CFG, "max_parallel_tasks", 1)
    monkeypatch.setitem(
        _dask_progress.CFG["logging"],
        "log_progress_interval",
        f"{interval}s" if interval > 0 else interval,
    )

    def func(delay: float) -> None:
        time.sleep(delay)

    delayeds = [dask.delayed(func)(0.11)]
    _dask_progress._compute_with_progress(delayeds, description="test")
    if interval == 0.0:  # noqa: SIM108
        # Assert that some progress bar has been written to stdout.
        progressbar = capsys.readouterr().out
    else:
        # Assert that some progress bar has been logged.
        progressbar = caplog.text
    if interval < 0.0:
        assert not progressbar
    else:
        assert "100%" in progressbar
    if client is not None:
        client.shutdown()
