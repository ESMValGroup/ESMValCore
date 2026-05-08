"""Tests for the (deprecated) esmvalcore.local module."""

from pathlib import Path

import pytest

from esmvalcore.exceptions import RecipeError
from esmvalcore.local import DataSource


def test_get_glob_patterns_missing_facets() -> None:
    """Test that get_glob_patterns raises when required facets are missing."""
    local_data_source = DataSource(
        name="test",
        project="test",
        priority=1,
        rootpath=Path("/climate_data"),
        dirname_template="{dataset}",
        filename_template="{short_name}*nc",
    )
    facets = {
        "short_name": "tas",
    }
    expected_message = (
        "Unable to complete path '{dataset}' because the facet 'dataset' has "
        "not been specified."
    )
    with pytest.raises(RecipeError, match=expected_message):
        local_data_source.get_glob_patterns(**facets)
