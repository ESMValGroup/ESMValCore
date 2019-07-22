"""Logging module for quicklook operations."""
from esmvalcore._config import get_config_user_file, get_ql_logger


def _ql_log(msg, log_level, *args):
    """Log to quicklook logger if enabled."""
    cfg = get_config_user_file()
    if cfg['quicklook']['active']:
        ql_logger = get_ql_logger()
        getattr(ql_logger, log_level)(msg, *args)


def ql_info(logger, msg, *args):
    """Log info to regular loggers and quicklook logger if enabled."""
    logger.info(msg, *args)
    _ql_log(msg, 'info', *args)


def ql_warning(logger, msg, *args):
    """Log warning to regular loggers and quicklook logger if enabled."""
    logger.warning(msg, *args)
    _ql_log(msg, 'warning', *args)


def ql_error(logger, msg, *args):
    """Log error to regular loggers and quicklook logger if enabled."""
    logger.error(msg, *args)
    _ql_log(msg, 'error', *args)
