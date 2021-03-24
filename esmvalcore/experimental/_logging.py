"""Logging utilities."""

import logging
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def log_to_dir(drc: Path):
    """Log messages to the specified directory.

    This is a context manager to temporarily redirect the logging when
    running a recipe. Messages will be printed to the console, as well as
    the specified directory. Handlers are attached to the root logger, and
    removed at the end of the block.

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

    # create file handler which logs even debug messages
    debug_log_file = logging.FileHandler(drc / 'main_log_debug.txt')
    debug_log_file.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s UTC [%(process)d] %(levelname)-7s'
        ' %(name)s:%(lineno)s %(message)s')
    debug_log_file.setFormatter(formatter)

    # create file handler which logs simple info messages
    simple_log_file = logging.FileHandler(drc / 'main_log.txt')
    simple_log_file.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)-7s [%(process)d] %(message)s')
    simple_log_file.setFormatter(formatter)

    # add the handlers to root logger
    logging.root.addHandler(debug_log_file)
    logging.root.addHandler(simple_log_file)

    yield

    logging.root.removeHandler(debug_log_file)
    logging.root.removeHandler(simple_log_file)
