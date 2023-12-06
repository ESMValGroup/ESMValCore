from pathlib import Path

import pytest

from esmvalcore.local import LocalFile, _path2facets


@pytest.mark.parametrize(
    'path,drs,expected',
    [
        (
            '/climate_data/value1/value2/filename.nc',
            '{facet1}/{facet2.lower}',
            {
                'facet1': 'value1',
                'facet2': 'value2',
            },
        ),
        (
            '/climate_data/value1/value1-value2/filename.nc',
            '{facet1}/{facet1}-{facet2}',
            {
                'facet1': 'value1',
                'facet2': 'value2',
            },
        ),
        (
            '/climate_data/value-1/value-1-value-2/filename.nc',
            '{facet1}/{facet1}-{facet2}',
            {
                'facet1': 'value-1',
                'facet2': 'value-2',
            },
        )
    ],
)
def test_path2facets(path, drs, expected):
    """Test `_path2facets."""
    filepath = Path(path)
    result = _path2facets(filepath, drs)

    assert result == expected


def test_localfile():
    file = LocalFile('/a/b.nc')
    file.facets = {'a': 'A'}
    assert Path(file) == Path('/a/b.nc')
    assert file.facets == {'a': 'A'}
