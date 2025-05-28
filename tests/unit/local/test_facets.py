from pathlib import Path

import pytest

from esmvalcore.local import DataSource, LocalFile


@pytest.mark.parametrize(
    ("path", "rootpath", "dirname_template", "filename_template", "facets"),
    [
        (
            "/climate_data/filename.nc",
            "/climate_data",
            "/",
            "*.nc",
            {},
        ),
        (
            "/climate_data/value1/filename.nc",
            "/climate_data",
            "{facet1}",
            "{facet2}.nc",
            {
                "facet1": "value1",
                "facet2": "filename",
            },
        ),
        (
            "/climate_data/value1/xyz/filename.nc",
            "/climate_data",
            "{facet1}/xyz",
            "{facet2}.nc",
            {
                "facet1": "value1",
                "facet2": "filename",
            },
        ),
        (
            "/climate_data/value1/value1/value1.nc",
            "/climate_data",
            "{facet1}/{facet1}",
            "{facet1}.nc",
            {
                "facet1": "value1",
            },
        ),
        (
            "/climate_data/value1/value1/value1.nc",
            "/climate_data",
            "{facet1.lower}/{facet1.upper}",
            "{facet1}.nc",
            {
                "facet1": "value1",
            },
        ),
        (
            "/climate_data/value1/value2/filename.nc",
            "/climate_data",
            "{facet1}/{facet2}",
            "*.nc",
            {
                "facet1": "value1",
                "facet2": "value2",
            },
        ),
        (
            "/climate_data/value1/xyz/value2/filename.nc",
            "/climate_data",
            "{facet1}/*/{facet2}",
            "*",
            {
                "facet1": "value1",
                "facet2": "value2",
            },
        ),
        (
            "/climate_data/value1/other/paths/value2/filename.nc",
            "/climate_data",
            "{facet1}/*/{facet2}",
            "{facet3}.nc",
            {
                "facet1": "value1",
                "facet2": "value2",
                "facet3": "filename",
            },
        ),
        (
            "/climate_data/value1/value2/value3/value4/filename.nc",
            "/climate_data",
            "{facet1}/{facet2.upper}/*/{facet3}",
            "*.nc",
            {
                "facet1": "value1",
                "facet2": "value2",
                "facet3": "value4",
            },
        ),
        (
            "/climate_data/value1/other/paths/value2/filename_other.nc",
            "/climate_data",
            "{facet1}/*/{facet2}",
            "{facet3}[_.]*.nc",
            {
                "facet1": "value1",
                "facet2": "value2",
                "facet3": "filename",
            },
        ),
        (
            "/climate_data/value1/other/paths/value2/filename.other.nc",
            "/climate_data",
            "{facet1}/*/{facet2}",
            "{facet3}[_.]*.nc",
            {
                "facet1": "value1",
                "facet2": "value2",
                "facet3": "filename",
            },
        ),
        (
            "/climate_data/Tier3/ds/ds_1993.nc",
            "/climate_data",
            "Tier{tier}/{dataset}",
            "{dataset}[_.]*nc",
            {
                "tier": "3",
                "dataset": "ds",
            },
        ),
        (
            "/climate_data/Tier3/ds/tas_ds_Amon_1993.nc",
            "/climate_data",
            "Tier{tier}/{dataset}",
            "{short_name}_*.nc",
            {
                "tier": "3",
                "dataset": "ds",
                "short_name": "tas",
            },
        ),
        (
            "/climate_data/tas_ds_Amon_1993.nc",
            "/climate_data",
            "/",
            "{short_name}_*",
            {
                "short_name": "tas",
            },
        ),
        (
            "/climate_data/tas_ds.nc",
            "/climate_data",
            "/",
            "{short_name}_{dataset}[_.]*nc",
            {
                "short_name": "tas",
                "dataset": "ds",
            },
        ),
        (
            "/climate_data/tas_ds_Amon_1993.nc",
            "/climate_data",
            "/",
            "{short_name}_{dataset}[_.]*",
            {
                "short_name": "tas",
                "dataset": "ds",
            },
        ),
        (
            "/climate_data/tas_ad",
            "/climate_data",
            "/",
            "/{short_name}_[ab][cd]*",
            {
                "short_name": "tas",
            },
        ),
        (
            "/climate_data/value1/value2-value3/filename.nc",
            "/climate_data",
            "{facet1}/{facet2.upper}-{facet3}",
            "*.nc",
            {
                "facet1": "value1",
                "facet2": "value2",
                "facet3": "value3",
            },
        ),
        (
            "/climate_data/value1/value1-value2/filename.nc",
            "/climate_data",
            "{facet1}/{facet1}-{facet2}",
            "{facet3}.nc",
            {
                "facet1": "value1",
                "facet2": "value2",
                "facet3": "filename",
            },
        ),
        (
            "/climate_data/value-1/value-1-value-2/filename.nc",
            "/climate_data",
            "{facet1}/{facet1}-{facet2}",
            "{facet3}.nc",
            {
                "facet1": "value-1",
                "facet2": "value-2",
                "facet3": "filename",
            },
        ),
        (
            "/climate_data/value-1-value-2/value-2/filename.nc",
            "/climate_data",
            "{facet1}-{facet2}/{facet2}",
            "{facet3}.nc",
            {
                "facet1": "value-1",
                "facet2": "value-2",
                "facet3": "filename",
            },
        ),
        (
            "/climate_data/value1/value2value3/filename.nc",
            "/climate_data",
            "{facet1}/{facet2}{facet3}",
            "*.nc",
            {
                "facet1": "value1",
            },
        ),
        (
            "/climate_data/value1/value2value3value4/filename.nc",
            "/climate_data",
            "{facet1}/{facet2}{facet3}{facet4}",
            "*.nc",
            {
                "facet1": "value1",
            },
        ),
        (
            "/climate_data/1/2345/678/910/11/filename.nc",
            "/climate_data",
            "{f1}/{f2}{f3}{f4}{f5}/{f6}{f7}{f8}/{f9}{f10}/{f11}",
            "{filename}.nc",
            {
                "f1": "1",
                "f11": "11",
                "filename": "filename",
            },
        ),
    ],
)
def test_path2facets(
    path,
    rootpath,
    dirname_template,
    filename_template,
    facets,
):
    """Test `DataSource.path2facets."""
    path = Path(path)
    rootpath = Path(rootpath)
    data_source = DataSource(rootpath, dirname_template, filename_template)
    result = data_source.path2facets(path)
    assert result == facets


def test_localfile():
    file = LocalFile("/a/b.nc")
    file.facets = {"a": "A"}
    assert Path(file) == Path("/a/b.nc")
    assert file.facets == {"a": "A"}
