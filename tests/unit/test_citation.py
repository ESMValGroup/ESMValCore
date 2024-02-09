from esmvalcore import _citation


def test_extract_tags():
    tags = "['example1', 'example_2', 'example-3']"
    result = _citation._extract_tags(tags)
    assert result == {'example1', 'example_2', 'example-3'}
