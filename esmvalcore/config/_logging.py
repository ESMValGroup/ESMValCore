"""Configure logging."""

import logging
import logging.config
import os
import time
from pathlib import Path
from typing import Union

import yaml


def _purge_file_handlers(cfg: dict) -> None:
    """Remove handlers with filename set.

    This is used to remove file handlers which require an output
    directory to be set.
    """
    cfg['handlers'] = {
        name: handler
        for name, handler in cfg['handlers'].items()
        if 'filename' not in handler
    }
    prev_root = cfg['root']['handlers']
    cfg['root']['handlers'] = [
        name for name in prev_root if name in cfg['handlers']
    ]


def _get_log_files(cfg: dict,
                   output_dir: Union[os.PathLike, str] = None) -> list:
    """Initialize log files for the file handlers."""
    log_files = []

    handlers = cfg['handlers']

    for handler in handlers.values():
        filename = handler.get('filename', None)

        if filename:
            if output_dir is None:
                raise ValueError('`output_dir` must be defined')

            if not os.path.isabs(filename):
                handler['filename'] = os.path.join(output_dir, filename)

            log_files.append(handler['filename'])

    return log_files


def _update_stream_level(cfg: dict, level=None):
    """Update the log level for the stream handlers."""
    handlers = cfg['handlers']

    for handler in handlers.values():
        if level is not None and 'stream' in handler:
            if handler['stream'] in ('ext://sys.stdout', 'ext://sys.stderr'):
                handler['level'] = level.upper()


def configure_logging(cfg_file: Union[os.PathLike, str] = None,
                      output_dir: Union[os.PathLike, str] = None,
                      console_log_level: str = None) -> list:
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
        cfg_file = Path(__file__).parent / 'config-logging.yml'

    cfg_file = Path(cfg_file).absolute()

    with open(cfg_file) as file_handler:
        cfg = yaml.safe_load(file_handler)

    if output_dir is None:
        _purge_file_handlers(cfg)

    log_files = _get_log_files(cfg, output_dir=output_dir)
    _update_stream_level(cfg, level=console_log_level)

    logging.config.dictConfig(cfg)
    logging.Formatter.converter = time.gmtime
    logging.captureWarnings(True)

    return log_files
