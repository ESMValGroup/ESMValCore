"""Test :mod:`esmvalcore.preprocessor._dask_progress`."""

from __future__ import annotations

import contextlib
import io
import time
from typing import TYPE_CHECKING

import dask
import distributed
import pytest
import rich.console
from loguru import logger

from esmvalcore.config._logging import _log_task, configure_logging
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
    configure_logging(console_log_level=session["log_level"])

    def func(delay: float) -> None:
        time.sleep(delay)

    delayeds = [dask.delayed(func)(0.11)]
    _dask_progress._compute_with_progress(
        delayeds,
        session=session,
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


@pytest.mark.parametrize("task", [None, "diagnostic/variable"])
def test_rich_progress_aligned(
    capsys: pytest.CaptureFixture,
    task: str | None,
) -> None:
    """Test that the Rich progress bars line up with the log messages."""
    configure_logging(console_log_level="info")
    console = rich.console.Console(
        file=io.StringIO(),
        width=200,
        color_system=None,
    )
    named = logger.patch(lambda r: r.update(name="esmvalcore.test"))
    with _log_task(task) if task else contextlib.nullcontext():
        named.info("message")
        progress = _dask_progress._rich_progress(console=console)
    progress.add_task("progress", total=1)
    console.print(progress.make_tasks_table(progress.tasks))
    log_line = capsys.readouterr().out
    bar_line = console.file.getvalue()
    start = log_line.index("message")
    # Skip the time, which may differ by a second.
    assert (
        bar_line[len("HH:mm:ss") : start] == log_line[len("HH:mm:ss") : start]
    )
    assert bar_line[start:].startswith("  0%")
