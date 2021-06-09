"""Tests for _replace_tags in _data_finder.py."""

from esmvalcore._data_finder import _replace_tags

VARIABLE = {
    'short_name': 'tas',
}


def test_replace_tags_str():
    assert _replace_tags('folder/subfolder/{short_name}',
                         VARIABLE) == ['folder/subfolder/tas']


def test_replace_tags_list_of_str():
    assert _replace_tags(('folder/subfolder/{short_name}',
                          'folder2/{short_name}', 'subfolder/{short_name}'),
                         VARIABLE) == [
                             'folder/subfolder/tas',
                             'folder2/tas',
                             'subfolder/tas',
                         ]
