"""Logging module for quicklook operations."""
import logging

from esmvalcore._config import get_config_user_file


def setup_quicklook_logger():
    """Initialize quicklook logger instance."""
    cfg = get_config_user_file()
    formatter = logging.Formatter(
        '%(asctime)s UTC [%(process)d] %(levelname)-7s %(message)s')
    # handler = logging.FileHandler(log_file)
    # handler.setFormatter(formatter)
    #
    # logger = logging.getLogger(name)
    # logger.setLevel(level)
    # logger.addHandler(handler)
    #
    # return logger
