from pathlib import Path

import pytest

from esmvalcore.local import LocalFile, _path2facets, _str2facets


@pytest.mark.parametrize(
    'path,template,facets',
    [
        (
            "value1/value2",
            "{facet1}/{facet2.lower}",
            {
                'facet1': 'value1',
            },
        ),
        (
            "Tier3/dataset1",
            "Tier{tier}/{dataset}",
            {
                'tier': '3',
                'dataset': 'dataset1',
            },
        ),
        (
            "value1/xyz/value2",
            "{facet1}/xyz/{facet2}",
            {
                'facet1': 'value1',
                'facet2': 'value2',
            },
        ),
        ("value1/value2value3", "{facet1}/{facet2}{facet3}", {
            'facet1': 'value1',
        }),
        (
            "value1/value2",
            "",
            {},
        ),
        # Handling of the cases below can be made smarter if needed in the
        # future.
        (
            "value1_value2.nc",
            "{facet1}_{facet2}[_.]*nc",
            {
                'facet1': 'value1',
            },
        ),
        (
            "value1/value2/xyz/value3",
            "{facet1}/{facet2}/*/{facet3}",
            {
                'facet1': 'value1',
            },
        ),
        (
            "value1/value2value3/value4",
            "{facet1}/{facet2}{facet3}/{facet4}",
            {
                'facet1': 'value1',
            },
        ),
    ])
def test_str2facets(path, template, facets):
    """Test `_str2facets."""
    result = _str2facets(path, template)
    assert result == facets


def test_path2facets():
    file = Path('/climate_data/value1/value2/value2_2000-2001.nc')
    facets = _path2facets(
        file,
        dirname_template='{facet1}/{facet2}',
        filename_template='{facet2}_*.nc',
    )
    assert facets == {'facet1': 'value1', 'facet2': 'value2'}


def test_localfile():
    file = LocalFile('/a/b.nc')
    file.facets = {'a': 'A'}
    assert Path(file) == Path('/a/b.nc')
    assert file.facets == {'a': 'A'}
