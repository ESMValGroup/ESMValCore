"""Configure logging."""

import os
import sys
import warnings
from functools import partial
from typing import Optional, Union

from iris.warnings import IrisUserWarning
from loguru import logger


def log_warnings(message, *args, **kwargs):
    """Redirect warnings to loguru logger."""
    # https://loguru.readthedocs.io/en/stable/resources/recipes.html#capturing-standard-stdout-stderr-and-warnings
    if isinstance(message, IrisUserWarning):
        # Iris does not correctly set the stack level for its warnings,
        # so increase the depth by 1 to get the relevant location of the
        # calling code.
        logger.opt(depth=3).debug("{}: {}", type(message).__name__, message)
    else:
        logger.opt(depth=2).warning(message)


def filter_by_name(record: dict, names: set[str], keep: bool = True) -> bool:
    return (record["name"] in names) == keep


def configure_logging(
    cfg_file: Optional[Union[os.PathLike, str]] = None,
    output_dir: Optional[Union[os.PathLike, str]] = None,
    console_log_level: str = "INFO",
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
    log_level = console_log_level.upper()
    logger.level("INFO", color="")
    logger.level("DEBUG", color="<blue>")
    debug_fmt = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS zz}</green> [{process}] <level>{level: <7}</level> <cyan>{name}</cyan>:<cyan>{line}</cyan>:<cyan>{function}</cyan> <level>{message}</level>"
    console_fmt = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS zz}</green> [{process}] <level>{level: <7}</level> <level>{message}</level>"
    cmor_modules = {
        "esmvalcore.cmor.check",
        "esmvalcore.cmor._fixes.fix.genericfix",
    }
    cmor = partial(filter_by_name, names=cmor_modules, keep=True)
    no_cmor = partial(filter_by_name, names=cmor_modules, keep=False)
    logger.remove()
    logger.add(
        sys.stdout,
        level=log_level,
        format=console_fmt,
        filter=no_cmor,
        enqueue=True,
    )
    if output_dir is None:
        log_files = []
    else:
        main_log = os.path.join(output_dir, "main_log.txt")
        debug_log = os.path.join(output_dir, "main_log_debug.txt")
        cmor_log = os.path.join(output_dir, "cmor_log.txt")
        log_files = [main_log, debug_log, cmor_log]
        logger.add(
            main_log,
            level=log_level,
            format=console_fmt,
            filter=no_cmor,
            enqueue=True,
        )
        logger.add(
            debug_log,
            level="DEBUG",
            format=debug_fmt,
            filter=no_cmor,
            enqueue=True,
        )
        logger.add(
            cmor_log,
            level="DEBUG",
            format=console_fmt,
            filter=cmor,
            enqueue=True,
        )

    # Patch the warnings module so it writes warnings to the loguru logger.
    warnings.showwarning = log_warnings

    return log_files
