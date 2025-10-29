from pathlib import Path

import pytest

from esmvalcore.config import CFG
from esmvalcore.config._config_validators import validate_config_developer
from esmvalcore.local import DataSource, _get_data_sources


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
    assert isinstance(source, DataSource)
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
