"""Configure logging."""

import logging
import logging.config
import os
import time
import warnings
from collections.abc import Iterable
from copy import copy
from pathlib import Path
from typing import Literal, Optional, Union

import yaml

from esmvalcore.exceptions import ESMValCoreUserWarning

# Unique ID to distinguish ESMValCore warnings from other warnings
ESMVALCORE_WARNING_ID = "E741CF251D2FD29FEACBFD591FE6EC06"


class FilterMultipleNames:
    """Only allow/disallow events from loggers with specific names."""

    def __init__(
        self,
        names: Iterable[str],
        mode: Literal["allow", "disallow"],
    ) -> None:
        """Initialize filter."""
        self.names = names
        self.starts_with_name = mode == "allow"

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter events."""
        for name in self.names:
            if record.name.startswith(name):
                return self.starts_with_name
        return not self.starts_with_name


class FilterExternalWarnings:
    """Do not show warnings from external packages."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter events."""
        apply_filter = (
            record.name != "py.warnings" or ESMVALCORE_WARNING_ID in record.msg
        )
        return apply_filter


class Formatter(logging.Formatter):
    """Format logging message (always remove ESMVALCORE_WARNING_ID)."""

    def format(self, record: logging.LogRecord) -> str:
        """Remove ESMVALCORE_WARNING_ID before default formatting."""
        record = copy(record)
        record.msg = record.msg.replace(ESMVALCORE_WARNING_ID, "")
        return super().format(record)


def _purge_file_handlers(cfg: dict) -> None:
    """Remove handlers with filename set.

    This is used to remove file handlers which require an output
    directory to be set.
    """
    cfg["handlers"] = {
        name: handler
        for name, handler in cfg["handlers"].items()
        if "filename" not in handler
    }
    prev_root = cfg["root"]["handlers"]
    cfg["root"]["handlers"] = [
        name for name in prev_root if name in cfg["handlers"]
    ]


def _get_log_files(
    cfg: dict,
    output_dir: Optional[Union[os.PathLike, str]] = None,
) -> list:
    """Initialize log files for the file handlers."""
    log_files = []

    handlers = cfg["handlers"]

    for handler in handlers.values():
        filename = handler.get("filename", None)

        if filename:
            if output_dir is None:
                raise ValueError("`output_dir` must be defined")

            if not os.path.isabs(filename):
                handler["filename"] = os.path.join(output_dir, filename)

            log_files.append(handler["filename"])

    return log_files


def _update_stream_level(cfg: dict, level=None):
    """Update the log level for the stream handlers."""
    handlers = cfg["handlers"]

    for handler in handlers.values():
        if level is not None and "stream" in handler:
            if handler["stream"] in ("ext://sys.stdout", "ext://sys.stderr"):
                handler["level"] = level.upper()


def configure_logging(
    cfg_file: Optional[Union[os.PathLike, str]] = None,
    output_dir: Optional[Union[os.PathLike, str]] = None,
    console_log_level: Optional[str] = None,
) -> list:
    """Configure logging.

    Parameters
    ----------
    cfg_file : str, optional
        Logging config file. If `None`, defaults to `configure-logging.yml`
    output_dir : str, optional
        Output directory for the log files. If `None`, log only to the console.
    console_log_level : str, optional
        If `None`, use the default (INFO).

    Returns
    -------
    log_files : list
        Filenames that will be logged to.
    """
    if cfg_file is None:
        cfg_file = Path(__file__).parent / "config-logging.yml"

    cfg_file = Path(cfg_file).absolute()

    with open(cfg_file, "r", encoding="utf-8") as file_handler:
        cfg = yaml.safe_load(file_handler)

    if output_dir is None:
        _purge_file_handlers(cfg)

    log_files = _get_log_files(cfg, output_dir=output_dir)
    _update_stream_level(cfg, level=console_log_level)

    logging.config.dictConfig(cfg)
    logging.Formatter.converter = time.gmtime
    logging.captureWarnings(True)

    # Add unique ID to ESMValCore warnings to be able to filter them during
    # logging
    original_showwarning = copy(warnings.showwarning)

    def showwarning(message, category, filename, lineno, file=None, line=None):
        """Add unique ID to ESMValCore warnings."""
        if issubclass(category, ESMValCoreUserWarning):
            if isinstance(message, str):
                message = ESMVALCORE_WARNING_ID + message
            else:
                message.args = (
                    ESMVALCORE_WARNING_ID + message.args[0],
                    *message.args[1:],
                )
        original_showwarning(message, category, filename, lineno, file, line)

    warnings.showwarning = showwarning

    return log_files
