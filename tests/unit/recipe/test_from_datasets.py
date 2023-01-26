from esmvalcore._recipe.from_datasets import _to_frozen


def test_to_frozen():
    data = {
        'abc': 'x',
        'a': {
            'b': [
                'd',
                'c',
            ],
        },
    }

    result = _to_frozen(data)
    expected = (
        (
            'a',
            ((
                'b',
                (
                    'c',
                    'd',
                ),
            ), ),
        ),
        ('abc', 'x'),
    )

    assert result == expected
