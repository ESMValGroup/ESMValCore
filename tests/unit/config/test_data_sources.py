from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import esmvalcore.cmor.table
import esmvalcore.config._data_sources
import esmvalcore.local
from esmvalcore.exceptions import InvalidConfigParameter

if TYPE_CHECKING:
    from esmvalcore.config import Session


def test_load_data_sources_no_project_data_sources_configured(
    session: Session,
) -> None:
    """Test that loading data sources when no data sources are configured raises."""
    with pytest.raises(
        InvalidConfigParameter,
        match=r"No data sources found for project 'test'.*",
    ):
        esmvalcore.config._data_sources._get_data_sources(
            session,
            project="test",
        )


@pytest.mark.parametrize("search_esgf", ["never", "when_missing", "always"])
def test_load_legacy_data_sources(
    monkeypatch: pytest.MonkeyPatch,
    session: Session,
    search_esgf: str,
) -> None:
    """Test that loading legacy data sources works."""
    for project in session["projects"]:
        session["projects"][project].pop("data", None)
    session["search_esgf"] = search_esgf
    session["download_dir"] = "~/climate_data"
    monkeypatch.setattr(esmvalcore.cmor.table, "CMOR_TABLES", {})
    monkeypatch.setitem(
        esmvalcore.local.CFG,
        "config_developer_file",
        Path(esmvalcore.__path__[0], "config-developer.yml"),
    )
    monkeypatch.setitem(
        esmvalcore.local.CFG,
        "rootpath",
        {"default": "~/climate_data"},
    )
    data_sources = esmvalcore.config._data_sources._get_data_sources(
        session,
        project="CMIP6",
    )
    assert len(data_sources) == 1 if search_esgf == "never" else 2
