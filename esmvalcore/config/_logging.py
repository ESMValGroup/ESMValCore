"""Configure logging."""

from __future__ import annotations

import contextlib
import contextvars
import datetime as dt
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
    from collections.abc import Generator, Iterable

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


_TASK: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "esmvalcore_task",
    default=None,
)
"""Name of the task that is currently running."""


@contextlib.contextmanager
def _log_task(name: str) -> Generator[None]:
    """Show the task name with the messages logged inside this block."""
    token = _TASK.set(name)
    try:
        with logger.contextualize(task=name):
            yield
    finally:
        _TASK.reset(token)


def _console_prefix() -> tuple[str, str | None]:
    """Get the start of a console line, for output that is not logged.

    Returns
    -------
    :
        The time (UTC) and the INFO level, formatted like the log messages,
        and the name of the current task, if any.
    """
    time = dt.datetime.now(dt.UTC).strftime("%H:%M:%S")
    return f"{time} {'INFO':<7}", _TASK.get()


def _console_format(record: Record) -> str:
    """Format messages for the console and the main log file.

    Messages show the time (UTC), the level, and the message, prefixed with
    the name of the task that logged it, if any. Continuation lines of
    multi-line messages are indented to the start of the message.
    """
    task = record["extra"].get("task")
    prefix = f"{task}: " if task else ""
    time_width = len("HH:mm:ss ")
    level_width = max(7, len(record["level"].name)) + 1
    indent = " " * (time_width + level_width + len(prefix))
    record["extra"]["_console_task"] = prefix
    if "\n" in record["message"]:
        first, *rest = record["message"].split("\n")
        # Do not indent empty lines, to avoid trailing whitespace.
        record["extra"]["_console_message"] = "\n".join(
            [first, *(indent + line if line else line for line in rest)],
        )
        message = "{extra[_console_message]}"
    else:
        # Use the message field when possible, because loguru only shows the
        # colors of messages logged with `logger.opt(colors=True)` there.
        message = "{message}"
    return (
        "{time:HH:mm:ss!UTC} <level>{level: <7}</level> "
        "<cyan>{extra[_console_task]}</cyan>"
        f"<level>{message}</level>\n"
        "{exception}"
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
    "main_log.txt": ("INFO", _console_format, _main_log_filter),
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
    # Only use colors for the less common levels, so they stand out.
    logger.level("INFO", color="")
    logger.level("DEBUG", color="<light-blue>")

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
        format=_console_format,
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
