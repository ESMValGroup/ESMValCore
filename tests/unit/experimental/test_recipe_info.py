from pathlib import Path

import esmvalcore
from esmvalcore._config._diagnostics import Diagnostics, TagsManager
from esmvalcore.experimental.recipe_info import Contributor, Project, Reference

TAGS = TagsManager({
    'authors': {
        'doe_john': {
            'name': 'Doe, John',
            'institute': 'Testing Institute',
            'orcid': 'https://orcid.org/0000-0002-6887-4885',
        }
    },
    'projects': {
        'test_project': 'Test Project',
    }
})

DIAGNOSTICS = Diagnostics(Path(__file__).parent)


def test_contributor(monkeypatch):
    """Coverage test for Contributor."""
    monkeypatch.setattr(esmvalcore.experimental.recipe_metadata, 'TAGS', TAGS)

    contributor = Contributor.from_tag('doe_john')

    assert contributor.name == 'John Doe'
    assert contributor.institute == 'Testing Institute'
    assert contributor.orcid.startswith('https://orcid.org/')
    assert isinstance(repr(contributor), str)
    assert isinstance(str(contributor), str)


def test_contributor_from_dict():
    """Test Contributor init from dict."""
    name = 'John Doe'
    institute = 'Institute'
    orcid = 'https://orcid.org/0000'
    attributes = {'name': name, 'institute': institute, 'orcid': orcid}
    author = Contributor.from_dict(attributes=attributes)
    assert author.name == name
    assert author.institute == institute
    assert author.orcid == orcid


def test_reference(monkeypatch):
    """Coverage test for Reference."""
    monkeypatch.setattr(esmvalcore.experimental.recipe_metadata, 'DIAGNOSTICS',
                        DIAGNOSTICS)

    reference = Reference.from_tag('doe2021')

    assert isinstance(repr(reference), str)
    assert isinstance(str(reference), str)
    assert isinstance(reference.render('markdown'), str)

    assert str(reference) == 'J. Doe. Test free or fail hard. 2021. doi:0.'


def test_project(monkeypatch):
    """Coverage test for Project."""
    monkeypatch.setattr(esmvalcore.experimental.recipe_metadata, 'TAGS', TAGS)

    project = Project.from_tag('test_project')

    assert isinstance(repr(project), str)
    assert isinstance(str(project), str)
    assert project.project == 'Test Project'
