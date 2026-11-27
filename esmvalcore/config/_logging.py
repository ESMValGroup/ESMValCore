"""Configure logging."""

from __future__ import annotations

import contextlib
import inspect
import logging
import sys
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

import esmvalcore.exceptions

if TYPE_CHECKING:
    import os
    from collections.abc import Iterable

    from loguru import Record

_WARNINGS_SHOWN_IN_MAIN_LOG = [
    cls.__name__
    for cls in vars(esmvalcore.exceptions).values()
    if inspect.isclass(cls)
    and issubclass(cls, esmvalcore.exceptions.ESMValCoreUserWarning)
]

_HANDLER_IDS: list[int] = []
"""Loguru handlers added by :func:`configure_logging`."""

_OUR_NAMES = ("esmvalcore", "esmvaltool", "intake-esgf", "py.warnings")
_CMOR_NAMES = (
    "esmvalcore.cmor.check",
    "esmvalcore.cmor._fixes.fix.genericfix",
)

_CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss,SSS!UTC}</green> UTC [{process}] "
    "<level>{level: <7}</level> <level>{message}</level>"
)
_BRIEF_FORMAT = "{level: <7} [{process}] {message}"
_DEBUG_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss,SSS!UTC} UTC [{process}] {level: <7} "
    "{name}:{line} {message}"
)


def _has_name(record: Record, names: Iterable[str]) -> bool:
    """Check if the record comes from (a child of) one of the loggers."""
    return any(
        record["name"] == name or record["name"].startswith(f"{name}.")
        for name in names
    )


def _is_external_warning(record: Record) -> bool:
    """Check if the record is a warning issued by an external package."""
    if record["name"] != "py.warnings":
        return False
    return not any(
        f" {warning}: " in record["message"]
        for warning in _WARNINGS_SHOWN_IN_MAIN_LOG
    )


def _main_log_filter(record: Record) -> bool:
    """Filter for the console and the main log file."""
    return (
        _has_name(record, _OUR_NAMES)
        and not _has_name(record, _CMOR_NAMES)
        and not _has_name(record, ["intake-esgf"])
        and not _is_external_warning(record)
    )


def _debug_log_filter(record: Record) -> bool:
    """Filter for the debug log file."""
    return _has_name(record, _OUR_NAMES)


def _cmor_log_filter(record: Record) -> bool:
    """Filter for the CMOR log file."""
    return _has_name(record, _CMOR_NAMES)


_LOG_FILES = {
    # The main log file contains the same messages as the console.
    "main_log.txt": ("INFO", _CONSOLE_FORMAT, _main_log_filter),
    "main_log_debug.txt": ("DEBUG", _DEBUG_FORMAT, _debug_log_filter),
    "cmor_log.txt": ("INFO", _BRIEF_FORMAT, _cmor_log_filter),
}
"""Log files: their level, format, and filter."""


class _InterceptHandler(logging.Handler):
    """Send log records from the standard library logging module to loguru.

    See https://loguru.readthedocs.io/en/stable/overview.html#entirely-compatible-with-standard-logging
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        logger.patch(
            lambda r: r.update(name=record.name, line=record.lineno),
        ).opt(exception=record.exc_info).log(level, record.getMessage())


def _log_warning(
    message: Warning | str,
    category: type[Warning],
    filename: str,
    lineno: int,
    file: object = None,  # noqa: ARG001
    line: str | None = None,
) -> None:
    """Send warnings to loguru, like :func:`logging.captureWarnings`."""
    text = warnings.formatwarning(message, category, filename, lineno, line)
    logger.patch(
        lambda r: r.update(name="py.warnings", line=lineno),
    ).warning(text.rstrip())


def _add_log_files(
    output_dir: os.PathLike | str,
    filenames: Iterable[str] = _LOG_FILES,
) -> dict[str, int]:
    """Add handlers that write to log files in `output_dir`.

    Returns
    -------
    :
        A mapping from the path of each log file to its handler id.
    """
    handlers = {}
    for filename in filenames:
        level, fmt, filter_ = _LOG_FILES[filename]
        # Only show variable values in tracebacks in the debug log.
        diagnose = level == "DEBUG"
        path = str(Path(output_dir) / filename)
        handlers[path] = logger.add(
            path,
            level=level,
            format=fmt,
            filter=filter_,
            encoding="utf-8",
            mode="a",
            backtrace=diagnose,
            diagnose=diagnose,
        )
    return handlers


def _capture_external_messages() -> None:
    """Send messages from the standard library and warnings to loguru."""
    logger.enable("esmvalcore")
    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)
    warnings.showwarning = _log_warning


def configure_logging(
    output_dir: os.PathLike | str | None = None,
    console_log_level: str | None = None,
) -> list[str]:
    """Configure logging.

    Parameters
    ----------
    output_dir:
        Output directory for the log files. If `None`, log only to the console.
    console_log_level:
        If `None`, use the default (INFO).

    Returns
    -------
    :
        Filenames that will be logged to.
    """
    level = "INFO" if console_log_level is None else console_log_level.upper()
    logger.level(level)  # Raises ValueError for unknown levels.

    # Only remove the handlers added by this function and the default
    # loguru handler, so other handlers (e.g. from pytest) keep working.
    for handler_id in _HANDLER_IDS:
        logger.remove(handler_id)
    _HANDLER_IDS.clear()
    with contextlib.suppress(ValueError):
        logger.remove(0)

    handler_id = logger.add(
        sys.stdout,
        level=level,
        format=_CONSOLE_FORMAT,
        filter=_main_log_filter,
        backtrace=False,
        diagnose=False,
    )
    _HANDLER_IDS.append(handler_id)
    log_files = {}
    if output_dir is not None:
        log_files = _add_log_files(output_dir)
        _HANDLER_IDS.extend(log_files.values())

    _capture_external_messages()

    return list(log_files)
