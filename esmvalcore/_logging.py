"""Configure logging."""

import logging
import os
import time

import yaml


def configure_logging(cfg_file=None, output_dir=None, console_log_level=None):
    """Set up logging."""
    if cfg_file is None:
        cfg_file = os.path.join(os.path.dirname(__file__),
                                'config-logging.yml')

    cfg_file = os.path.abspath(cfg_file)
    with open(cfg_file) as file_handler:
        cfg = yaml.safe_load(file_handler)

    if output_dir is None:
        cfg['handlers'] = {
            name: handler
            for name, handler in cfg['handlers'].items()
            if 'filename' not in handler
        }
        prev_root = cfg['root']['handlers']
        cfg['root']['handlers'] = [
            name for name in prev_root if name in cfg['handlers']
        ]

    log_files = []
    for handler in cfg['handlers'].values():
        if 'filename' in handler:
            if not os.path.isabs(handler['filename']):
                handler['filename'] = os.path.join(output_dir,
                                                   handler['filename'])
            log_files.append(handler['filename'])
        if console_log_level is not None and 'stream' in handler:
            if handler['stream'] in ('ext://sys.stdout', 'ext://sys.stderr'):
                handler['level'] = console_log_level.upper()

    logging.config.dictConfig(cfg)
    logging.Formatter.converter = time.gmtime
    logging.captureWarnings(True)

    return log_files
