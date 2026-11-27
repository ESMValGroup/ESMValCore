"""Tests for :mod:`esmvalcore.experimental._logging`."""

import pytest
from loguru import logger

from esmvalcore.config._logging import _HANDLER_IDS, configure_logging
from esmvalcore.experimental._logging import _log_to_stdout, log_to_dir


def test_log_to_dir(tmp_path):
    """Test that messages are logged to files only inside the block."""
    named = logger.patch(lambda r: r.update(name="esmvalcore.test"))
    with log_to_dir(tmp_path / "run"):
        named.info("info inside")
        named.debug("debug inside")
    named.info("info after")

    main_log = (tmp_path / "run" / "main_log.txt").read_text(encoding="utf-8")
    debug_log = (tmp_path / "run" / "main_log_debug.txt").read_text(
        encoding="utf-8",
    )
    assert "info inside" in main_log
    assert "debug inside" not in main_log
    assert "info inside" in debug_log
    assert "debug inside" in debug_log
    assert "info after" not in main_log + debug_log
    assert not (tmp_path / "run" / "cmor_log.txt").exists()


@pytest.fixture
def unconfigured_logging():
    """Remove the handlers added by previous calls to `configure_logging`."""
    for handler_id in _HANDLER_IDS:
        logger.remove(handler_id)
    _HANDLER_IDS.clear()


@pytest.mark.parametrize("experimental_first", [True, False])
@pytest.mark.usefixtures("unconfigured_logging")
def test_log_to_stdout_once(capsys, experimental_first):
    """Test that messages are printed once when logging is configured."""
    if experimental_first:
        _log_to_stdout()
        configure_logging(console_log_level="info")
    else:
        configure_logging(console_log_level="info")
        _log_to_stdout()
    logger.patch(lambda r: r.update(name="esmvalcore.test")).info("message")
    assert capsys.readouterr().out.count("message") == 1
