"""Test :mod:`esmvalcore.preprocessor._dask_progress`."""

import logging
import operator

import dask
import distributed
import pytest

from esmvalcore.preprocessor import _dask_progress


@pytest.mark.parametrize("use_distributed", [False, True])
@pytest.mark.parametrize("ntasks", [1, 2])
def test_compute_with_progress(
    capsys,
    caplog,
    monkeypatch,
    use_distributed,
    ntasks,
):
    caplog.set_level(logging.INFO)
    if use_distributed:
        client = distributed.Client(n_workers=1, threads_per_worker=1)
    else:
        client = None

    monkeypatch.setitem(_dask_progress.CFG, "max_parallel_tasks", ntasks)
    delayeds = [dask.delayed(operator.add)(1, 1)]
    _dask_progress._compute_with_progress(delayeds, description="test")
    # Assert that some progress bar has been written to stdout.
    if ntasks == 1:
        progressbar = capsys.readouterr().out
    else:
        progressbar = caplog.text
    assert "100%" in progressbar
    if client is not None:
        client.shutdown()
