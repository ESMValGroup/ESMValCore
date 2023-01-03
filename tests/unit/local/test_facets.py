from pathlib import Path

from esmvalcore.local import LocalFile, _path2facets


def test_path2facets():
    """Test `_path2facets1."""
    filepath = Path("/climate_data/value1/value2/filename.nc")
    drs = "{facet1}/{facet2.lower}"

    expected = {
        'facet1': 'value1',
        'facet2': 'value2',
    }

    result = _path2facets(filepath, drs)

    assert result == expected


def test_localfile():
    file = LocalFile('/a/b.nc')
    file.facets = {'a': 'A'}
    assert Path(file) == Path('/a/b.nc')
    assert file.facets == {'a': 'A'}
