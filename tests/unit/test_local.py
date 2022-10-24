from pathlib import Path

from esmvalcore.local import LocalFile


def test_from_path():
    """Test `_path2facets1."""
    filepath = Path("/climate_data/value1/value2/filename.nc")
    drs = "{facet1}/{facet2.lower}"

    expected = {
        'facet1': 'value1',
        'facet2': 'value2',
    }

    file = LocalFile._from_path(filepath, drs, try_timerange=False)

    assert file.facets == expected
