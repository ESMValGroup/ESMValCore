import pytest

from esmvalcore.experimental import get_recipe
from esmvalcore.experimental.recipe import Contributor, Project, Reference

pytest.importorskip(
    'esmvaltool',
    reason='The behaviour of these tests depends on what ``DIAGNOSTICS.path``'
    'points to. This is defined by a forward-reference to ESMValTool, which'
    'is not installed in the CI, but likely to be available in a developer'
    'or user installation.')


def test_contributor():
    """Coverage test for Contributor."""
    contributor = Contributor.from_tag('righi_mattia')

    assert contributor.name == 'Mattia Righi'
    assert contributor.institute == 'DLR, Germany'
    assert contributor.orcid.startswith('https://orcid.org/')
    assert isinstance(repr(contributor), str)
    assert isinstance(str(contributor), str)


def test_reference():
    """Coverage test for Reference."""
    reference = Reference.from_tag('acknow_project')

    assert isinstance(repr(reference), str)
    assert isinstance(str(reference), str)
    assert isinstance(reference.render('markdown'), str)


def test_project():
    """Coverage test for Project."""
    project = Project.from_tag('esmval')

    assert isinstance(repr(project), str)
    assert isinstance(str(project), str)


def test_recipe():
    """Coverage test for Recipe."""
    recipe = get_recipe('examples/recipe_python')

    assert isinstance(repr(recipe), str)
    assert isinstance(str(recipe), str)
    assert isinstance(recipe.to_markdown(), str)
