"""Tests for logging config submodule."""

import logging
import re
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import pytest
from loguru import logger

from esmvalcore.config._logging import (
    _WARNINGS_SHOWN_IN_MAIN_LOG,
    _has_name,
    _InterceptHandler,
    _is_external_warning,
    configure_logging,
)
from esmvalcore.exceptions import ESMValCoreUserWarning


@pytest.mark.parametrize("level", [None, "INFO", "DEBUG"])
def test_logging_with_level(level):
    """Test log level configuration."""
    ret = configure_logging(console_log_level=level)
    assert isinstance(ret, list)
    assert len(ret) == 0


def test_logging_with_output_dir(tmp_path):
    """Test that the log files are written to the output directory."""
    ret = configure_logging(output_dir=tmp_path)
    assert isinstance(ret, list)
    assert len(ret) == 3
    for path in ret:
        assert tmp_path == Path(path).parent


def test_logging_log_level_invalid():
    """Test failure condition for invalid level specification."""
    with pytest.raises(ValueError, match="FAIL"):
        configure_logging(console_log_level="FAIL")


def test_log_files_content(tmp_path):
    """Test that messages end up in the right log files."""
    configure_logging(output_dir=tmp_path, console_log_level="info")
    logger.patch(lambda r: r.update(name="esmvalcore.test")).info("ours")
    logger.patch(lambda r: r.update(name="esmvalcore.test")).debug("debug")
    logger.patch(lambda r: r.update(name="esmvalcore.cmor.check")).info("c")
    logging.getLogger("intake-esgf").info("from intake-esgf")
    logging.getLogger("other").warning("from other package")
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.warn("external warning", UserWarning, stacklevel=1)
        warnings.warn("our warning", ESMValCoreUserWarning, stacklevel=1)
    logger.complete()
    configure_logging()  # Close the log files.

    main_log = (tmp_path / "main_log.txt").read_text(encoding="utf-8")
    debug_log = (tmp_path / "main_log_debug.txt").read_text(encoding="utf-8")
    cmor_log = (tmp_path / "cmor_log.txt").read_text(encoding="utf-8")

    assert "ours" in main_log
    assert "debug" not in main_log
    assert "from intake-esgf" not in main_log
    assert "external warning" not in main_log
    assert "our warning" in main_log
    assert "\nc\n" not in main_log

    assert "debug" in debug_log
    assert "from intake-esgf" in debug_log
    assert "external warning" in debug_log
    assert "from other package" not in debug_log

    assert cmor_log.strip().endswith(" c")


def test_intercept_handler_unknown_level():
    """Test that records with a level unknown to loguru are forwarded."""
    records = []
    handler_id = logger.add(records.append, level=0)
    record = logging.LogRecord(
        name="esmvalcore.test",
        level=15,
        pathname=__file__,
        lineno=42,
        msg="custom %s",
        args=("level",),
        exc_info=None,
    )
    try:
        _InterceptHandler().emit(record)
    finally:
        logger.remove(handler_id)
    (message,) = records
    assert message.record["level"].no == 15
    assert message.record["name"] == "esmvalcore.test"
    assert message.record["line"] == 42
    assert message.record["message"] == "custom level"


@pytest.mark.parametrize(
    ("names", "output"),
    [
        (["test"], False),
        (["test", "another.test"], False),
        (["test", "m.a.b.c"], False),
        (["a.b.c"], True),
        (["a"], True),
        (["a.b", "test"], True),
        (["a.b", "a.b.c"], True),
        (["a.b.c.d"], False),
        (["a.b.cd"], False),
    ],
)
def test_has_name(names, output):
    """Test `_has_name`."""
    assert _has_name({"name": "a.b.c"}, names) is output


@pytest.mark.parametrize(
    ("name", "msg", "output"),
    [
        ("test.module", "warning", False),
        ("test.module", "1: ESMValCoreUserWarning: warning", False),
        ("py.warnings", "warning", True),
        ("py.warnings", "1: ESMValCoreUserWarning: warning", False),
        ("py.warnings", "1: ESMValCoreDeprecationWarning: warning", False),
        ("py.warnings", "1: MissingConfigParameter: warning", False),
    ],
)
def test_is_external_warning(name, msg, output):
    """Test `_is_external_warning`."""
    assert _is_external_warning({"name": name, "message": msg}) is output


def test_warnings_shown_in_main_log():
    expected_warnings = {
        "ESMValCoreUserWarning",
        "ESMValCoreDeprecationWarning",
        "MissingConfigParameter",
        "ESMValCorePreprocessorWarning",
        "ESMValCoreLoadWarning",
    }
    assert len(_WARNINGS_SHOWN_IN_MAIN_LOG) == 5
    assert set(_WARNINGS_SHOWN_IN_MAIN_LOG) == expected_warnings


def test_no_messages_until_configured(tmp_path):
    """Test that the package does not log anything until configured."""
    code = textwrap.dedent(
        f"""
        from esmvalcore._task import write_ncl_settings
        from esmvalcore.config._logging import configure_logging

        write_ncl_settings({{}}, "{tmp_path / "before.ncl"}")
        configure_logging(console_log_level="debug")
        write_ncl_settings({{}}, "{tmp_path / "after.ncl"}")
        """,
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        check=True,
        text=True,
    )
    assert "before.ncl" not in result.stdout + result.stderr
    assert "after.ncl" in result.stdout


def test_main_log_same_as_console(tmp_path, capsys):
    """Test that the main log file looks the same as the console output."""
    configure_logging(output_dir=tmp_path, console_log_level="info")
    logger.patch(lambda r: r.update(name="esmvalcore.test")).info("message")
    configure_logging()  # Close the log files.
    console = capsys.readouterr().out
    main_log = (tmp_path / "main_log.txt").read_text(encoding="utf-8")
    assert re.fullmatch(
        r"\d\d:\d\d:\d\d INFO    message\n",
        main_log,
    )
    assert console == main_log


def test_console_format_task_and_indent(capsys):
    """Test that the task is shown and continuation lines are indented."""
    configure_logging(console_log_level="info")
    named = logger.patch(lambda r: r.update(name="esmvalcore.test"))
    with logger.contextualize(task="diag/variable"):
        named.info("first line\n\nthird line")
    with logger.contextualize(task="d/v"):
        named.warning("short task name")
    named.info("no task\ncontinued")
    lines = [
        line[len("HH:mm:ss ") :]
        for line in capsys.readouterr().out.splitlines()
    ]
    assert lines == [
        "INFO    diag/variable: first line",
        "",
        "                       third line",
        "WARNING d/v: short task name",
        "INFO    no task",
        "        continued",
    ]


def test_debug_log_task(tmp_path):
    """Test that the debug log shows the task like the main log."""
    configure_logging(output_dir=tmp_path, console_log_level="info")
    named = logger.patch(lambda r: r.update(name="esmvalcore.test"))
    with logger.contextualize(task="diag/variable"):
        named.info("with task")
    named.info("without task")
    configure_logging()  # Close the log files.
    debug_log = (tmp_path / "main_log_debug.txt").read_text(encoding="utf-8")
    lines = debug_log.splitlines()
    assert lines[0].endswith(" diag/variable: with task")
    assert re.search(r":\d+ without task$", lines[1])


def test_colored_message_without_terminal(tmp_path, capsys):
    """Test that color markup in messages is removed without a terminal."""
    configure_logging(output_dir=tmp_path, console_log_level="info")
    logger.patch(lambda r: r.update(name="esmvalcore.test")).opt(
        colors=True,
    ).info("Creating task <cyan>{}</cyan>", "a<b>/c{}")
    configure_logging()  # Close the log files.
    main_log = (tmp_path / "main_log.txt").read_text(encoding="utf-8")
    assert main_log.endswith(" INFO    Creating task a<b>/c{}\n")
    assert capsys.readouterr().out == main_log
