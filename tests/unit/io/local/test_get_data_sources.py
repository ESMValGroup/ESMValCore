from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from esmvalcore.config import CFG
from esmvalcore.config._config_validators import validate_config_developer
from esmvalcore.io.local import LocalDataSource
from esmvalcore.local import DataSource, _get_data_sources

if TYPE_CHECKING:
    import pytest_mock


@pytest.mark.parametrize(
    "rootpath_drs",
    [
        (
            {"CMIP6": {"/climate_data": "ESGF"}},
            {},
        ),
        (
            {"CMIP6": ["/climate_data"]},
            {"CMIP6": "ESGF"},
        ),
        (
            {"default": ["/climate_data"]},
            {"CMIP6": "ESGF"},
        ),
    ],
)
def test_get_data_sources(monkeypatch, rootpath_drs):
    # Make sure that default config-developer file is used
    validate_config_developer(None)

    rootpath, drs = rootpath_drs
    monkeypatch.setitem(CFG, "rootpath", rootpath)
    monkeypatch.setitem(CFG, "drs", drs)
    sources = _get_data_sources("CMIP6")
    source = sources[0]
    assert isinstance(source, LocalDataSource)
    assert source.rootpath == Path("/climate_data")
    assert "{project}" in source.dirname_template
    assert "{short_name}" in source.filename_template


def test_get_data_sources_nodefault(monkeypatch):
    # Make sure that default config-developer file is used
    validate_config_developer(None)

    monkeypatch.setitem(
        CFG,
        "rootpath",
        {
            "CMIP5": {"/climate_data": "default"},
        },
    )
    with pytest.raises(KeyError):
        _get_data_sources("CMIP6")


def test_data_source_deprecated(mocker: pytest_mock.MockerFixture) -> None:
    """Test that DataSource is deprecated."""
    mocker.patch.object(DataSource, "_path2facets")
    mocker.patch.object(DataSource, "find_data")
    with pytest.deprecated_call():
        data_source = DataSource(
            name="test",
            project="CMIP6",
            priority=1,
            rootpath=Path("/climate_data"),
            dirname_template="/",
            filename_template="*.nc",
        )

    assert data_source.regex_pattern
    assert data_source.get_glob_patterns() == [Path("/climate_data/*.nc")]
    data_source.path2facets(Path("/climate_data/some_file.nc"), False)
    data_source._path2facets.assert_called()  # type: ignore[attr-defined]
    data_source.find_files(dataset="a")
    data_source.find_data.assert_called()  # type: ignore[attr-defined]
