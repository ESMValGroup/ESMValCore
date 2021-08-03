"""Test 1esmvalcore.esgf._search`."""
from esmvalcore.esgf._search import expand_facets


def test_expand_facets():
    """Test that facets that are a tuple are correctly expanded."""
    facets = {
        'a': 1,
        'b': 2,
        'c': (3, 4),
        'd': (5, 6),
    }
    result = expand_facets(facets)

    reference = [
        {
            'a': 1,
            'b': 2,
            'c': 3,
            'd': 5,
        },
        {
            'a': 1,
            'b': 2,
            'c': 3,
            'd': 6,
        },
        {
            'a': 1,
            'b': 2,
            'c': 4,
            'd': 5,
        },
        {
            'a': 1,
            'b': 2,
            'c': 4,
            'd': 6,
        },
    ]
    assert result == reference
