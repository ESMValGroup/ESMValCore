"""Test that log messages from Dask workers end up in the log files."""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import pytest
from loguru import logger

from esmvalcore.config._dask import get_distributed_client
from esmvalcore.config._logging import configure_logging

if TYPE_CHECKING:
    from pathlib import Path

    from esmvalcore.config import Session


def _log_on_worker() -> int:
    """Log a message as if it came from ESMValCore."""
    pid = os.getpid()
    named = logger.patch(lambda r: r.update(name="esmvalcore.test"))
    named.debug("debug message from worker {}", pid)
    named.warning("warning from worker {}", pid)
    try:
        1 / 0  # noqa: B018
    except ZeroDivisionError:
        named.exception("exception on worker")
    return pid


def _read_when_logged(path: Path, text: str, timeout: float = 10.0) -> str:
    """Wait until `text` appears in `path` and return the file content."""
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        content = path.read_text(encoding="utf-8")
        if text in content:
            return content
        time.sleep(0.1)
    return path.read_text(encoding="utf-8")


@pytest.mark.parametrize("processes", [True, False])
def test_worker_messages_logged(
    tmp_path: Path,
    session: Session,
    processes: bool,
) -> None:
    session["dask"] = {
        "use": "local",
        "profiles": {
            "local": {
                "cluster": {
                    "type": "distributed.LocalCluster",
                    "n_workers": 1,
                    "threads_per_worker": 1,
                    "processes": processes,
                },
            },
        },
    }
    configure_logging(output_dir=tmp_path, console_log_level="info")
    with get_distributed_client(session) as client:
        assert client is not None
        pid = client.submit(_log_on_worker, pure=False).result()
        warning = f"warning from worker {pid}"
        debug_log = _read_when_logged(
            tmp_path / "main_log_debug.txt",
            "ZeroDivisionError",
        )
    configure_logging()  # Close the log files.
    main_log = (tmp_path / "main_log.txt").read_text(encoding="utf-8")

    assert (pid != os.getpid()) is processes
    assert debug_log.count(f"debug message from worker {pid}") == 1
    assert debug_log.count(warning) == 1
    assert "esmvalcore.test:" in debug_log
    assert "exception on worker\nTraceback" in debug_log
    assert "ZeroDivisionError: division by zero" in debug_log
    assert main_log.count(warning) == 1
    assert "debug message from worker" not in main_log
