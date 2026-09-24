from __future__ import annotations

from pathlib import Path

import pytest

import esmvalcore
import esmvalcore.cmor.table
from esmvalcore.config import CFG
from esmvalcore.io.local import LocalDataSource
from esmvalcore.local import _get_data_sources


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
    monkeypatch.setattr(esmvalcore.cmor.table, "CMOR_TABLES", {})
    monkeypatch.setitem(
        CFG,
        "config_developer_file",
        Path(esmvalcore.__path__[0], "config-developer.yml"),
    )

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
    monkeypatch.setattr(esmvalcore.cmor.table, "CMOR_TABLES", {})
    monkeypatch.setitem(
        CFG,
        "config_developer_file",
        Path(esmvalcore.__path__[0], "config-developer.yml"),
    )

    monkeypatch.setitem(
        CFG,
        "rootpath",
        {
            "CMIP5": {"/climate_data": "default"},
        },
    )
    with pytest.raises(KeyError):
        _get_data_sources("CMIP6")
