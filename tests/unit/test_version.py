"""Test that esmvalcore.__version__ returns a version number."""

from __future__ import annotations

import importlib
import re
from importlib.metadata import PackageNotFoundError
from typing import TYPE_CHECKING

import pytest

import esmvalcore
import esmvalcore._version

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def test_version():
    assert re.match(r"^\d+\.\d+\.\d+\S*$", esmvalcore.__version__)


def test_version_package_not_found_fail(mocker: MockerFixture) -> None:
    mocker.patch(
        "importlib.metadata.version",
        side_effect=PackageNotFoundError("ESMValCore"),
    )
    msg = r"ESMValCore package not found"
    with pytest.raises(PackageNotFoundError, match=re.escape(msg)):
        importlib.reload(esmvalcore._version)
