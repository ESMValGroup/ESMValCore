from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import esmvalcore.config._data_sources
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
