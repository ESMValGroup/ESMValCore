import textwrap
from pathlib import Path

import esmvalcore
from esmvalcore._config import TAGS
from esmvalcore._config._diagnostics import Diagnostics
from esmvalcore.experimental.recipe_info import (
    Contributor,
    Project,
    RecipeInfo,
    Reference,
)

DIAGNOSTICS = Diagnostics(Path(__file__).parent)


def test_contributor():
    """Coverage test for Contributor."""
    TAGS.set_tag_value(section='authors',
                       tag='doe_john',
                       value={
                           'name': 'Doe, John',
                           'institute': 'Testing',
                           'orcid': 'https://orcid.org/0000-0000-0000-0000',
                       })

    contributor = Contributor.from_tag('doe_john')

    assert contributor.name == 'John Doe'
    assert contributor.institute == 'Testing'
    assert contributor.orcid == 'https://orcid.org/0000-0000-0000-0000'
    assert isinstance(repr(contributor), str)
    assert isinstance(str(contributor), str)


def test_contributor_from_dict():
    """Test Contributor init from dict."""
    name = 'John Doe'
    institute = 'Testing'
    orcid = 'https://orcid.org/0000-0000-0000-0000'
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


def test_project():
    """Coverage test for Project."""
    TAGS.set_tag_value('projects', 'test_project', 'Test Project')

    project = Project.from_tag('test_project')

    assert isinstance(repr(project), str)
    assert isinstance(str(project), str)
    assert project.project == 'Test Project'


def test_recipe_info_str():
    """Test `RecipeInfo.__str__`."""
    data = {
        'documentation': {
            'title': 'Test recipe',
            'description': 'This is a very empty test recipe.'
        }
    }

    recipe = RecipeInfo(data, filename='/path/to/recipe_test.yml')

    text = textwrap.dedent("""
        ## Test recipe

        This is a very empty test recipe.

        ### Authors

        ### Maintainers
        """).lstrip()
    assert str(recipe) == text
