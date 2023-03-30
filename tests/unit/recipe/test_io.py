from esmvalcore._recipe import _io


def test_copy_dict():
    a = {'a': 1}
    b = {'b': a, 'c': a}
    result = _io._copy(b)
    assert result['b'] == result['c']
    assert result['b'] is not result['c']


def test_copy_list():
    a = ['a']
    b = {'b': a, 'c': a}
    result = _io._copy(b)
    assert result['b'] == result['c']
    assert result['b'] is not result['c']
