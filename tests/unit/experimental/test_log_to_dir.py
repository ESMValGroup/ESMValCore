"""Tests for :func:`esmvalcore.experimental._logging.log_to_dir`."""

from loguru import logger

from esmvalcore.experimental._logging import log_to_dir


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
