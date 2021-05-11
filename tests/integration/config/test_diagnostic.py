"""Test Diagnostics and TagsManager."""
import pytest

from esmvalcore._config._diagnostics import Diagnostics, TagsManager


def test_diagnostics_class():
    """Test diagnostics class and path locations."""
    diagnostics = Diagnostics.find()

    assert isinstance(diagnostics, Diagnostics)

    path = diagnostics.path

    assert diagnostics.recipes == path / 'recipes'
    assert diagnostics.references == path / 'references'
    assert diagnostics.tags_config == path / 'config-references.yml'
    assert diagnostics.scripts == path / 'diag_scripts'
    assert isinstance(diagnostics.load_tags(), TagsManager)


def test_tags_manager_setters():
    """Test TagsManager setters."""
    tags = TagsManager()
    tags.set_tag_value('section', 'tag1', 'value1')
    assert tags.get_tag_value('section', 'tag1') == 'value1'

    tags.set_tag_values({
        'section': {
            'tag2': 'value2',
        },
        'other': {
            'tag1': 'value1',
            'tag2': 'value2',
        },
    })

    assert tags.get_tag_value('section', 'tag1') == 'value1'
    assert tags.get_tag_value('section', 'tag2') == 'value2'
    assert tags.get_tag_value('other', 'tag1') == 'value1'
    assert tags.get_tag_value('other', 'tag2') == 'value2'


def test_tags_manager():
    """Test TagsManager functionality."""
    tags = TagsManager({'section': {'tag1': 123, 'tag2': 345}})

    ret = tags.get_tag_value('section', 'tag1')
    assert ret == 123

    ret = tags.get_tag_values('section', ('tag1', 'tag2'))
    assert ret == (123, 345)

    dict_with_tags = {'section': ['tag1', 'tag2']}
    tags.replace_tags_in_dict(dict_with_tags)

    assert dict_with_tags == {'section': (123, 345)}


def test_tags_manager_fails():
    """Test TagsManager fails."""
    tags = TagsManager({'section': {'tag1': 123, 'tag2': 345}})

    with pytest.raises(ValueError):
        tags.get_tag_value(section='undefined', tag='tag1')

    with pytest.raises(ValueError):
        tags.get_tag_value(section='section', tag='undefined')

    with pytest.raises(ValueError):
        dict_with_undefined_tags = {'section': ['tag1', 'undefined']}
        tags.replace_tags_in_dict(dict_with_undefined_tags)


def test_load_tags_from_non_existant_file():
    """Test fallback if no diagnostics are installed."""
    tags = TagsManager.from_file('non-existent')
    assert isinstance(tags, TagsManager)
    assert tags == {}
