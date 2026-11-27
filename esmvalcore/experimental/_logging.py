"""Logging utilities."""

from __future__ import annotations

import contextlib
import sys
from contextlib import contextmanager
from typing import TYPE_CHECKING

from loguru import logger

from esmvalcore.config._logging import (
    _add_log_files,
    _capture_external_messages,
    _main_log_filter,
)

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path


def _log_to_stdout() -> None:
    """Print informative log messages to stdout.

    This replaces the default loguru handler, which prints messages of all
    levels to stderr.
    """
    with contextlib.suppress(ValueError):
        logger.remove(0)
    logger.add(
        sys.stdout,
        level="INFO",
        format="{message}",
        filter=_main_log_filter,
        backtrace=False,
        diagnose=False,
    )
    _capture_external_messages()


@contextmanager
def log_to_dir(drc: Path) -> Generator[None]:
    """Log messages to the specified directory.

    This is a context manager to temporarily redirect the logging when
    running a recipe. Messages will be printed to the console, as well as
    the specified directory. The log files are closed at the end of the block.

    Messages will be logged to `<drc>/main_log.txt` and
    `<drc>/main_log_debug.txt`.

    .. code-block python

        with log_to_dir(session.run_dir):
            logger.info('/home/user/some/path')

    Parameters
    ----------
    drc : str
        Location where the logs should be stored.
    """
    drc.mkdir(parents=True, exist_ok=True)
    handlers = _add_log_files(drc, ["main_log.txt", "main_log_debug.txt"])
    try:
        yield
    finally:
        for handler_id in handlers.values():
            logger.remove(handler_id)
