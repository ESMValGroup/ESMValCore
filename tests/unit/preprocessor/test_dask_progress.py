"""Test :mod:`esmvalcore.preprocessor._dask_progress`."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import dask
import distributed
import pytest

from esmvalcore.preprocessor import _dask_progress

if TYPE_CHECKING:
    from esmvalcore.config import Session


@pytest.mark.parametrize("use_distributed", [False, True])
@pytest.mark.parametrize("interval", [-1, 0.0, 0.2])
def test_compute_with_progress(
    capsys: pytest.CaptureFixture,
    session: Session,
    use_distributed: bool,
    interval: float,
) -> None:
    if use_distributed:
        client = distributed.Client(n_workers=1, threads_per_worker=1)
    else:
        client = None

    session["log_level"] = "INFO"
    session["max_parallel_tasks"] = 1
    session["logging"]["log_progress_interval"] = (
        f"{interval}s" if interval > 0 else interval
    )
    session.run_dir.mkdir(parents=True)

    def func(delay: float) -> None:
        time.sleep(delay)

    delayeds = [dask.delayed(func)(0.11)]
    _dask_progress._compute_with_progress(
        delayeds,
        session=session,
        description="test",
    )
    progressbar = capsys.readouterr().out
    print(progressbar)
    if interval < 0.0:
        assert not progressbar
    else:
        assert "100%" in progressbar
        if interval == 0.0:
            # Assert that Rich progress bar has been written to stdout.
            assert "1/1" in progressbar
        else:
            # Assert that progress bar has been logged.
            assert "####" in progressbar
    if client is not None:
        client.shutdown()
