"""Unit tests for :func:`esmvalcore.recipe.format_recipe`."""

from __future__ import annotations

import textwrap

import pytest
import yaml

from esmvalcore.recipe import format_recipe, to_yaml

RECIPE = textwrap.dedent("""
    # A comment that is removed.
    # Quotes are only kept where they are needed.
    ---
    diagnostics:
      diag1:
        variables:
          tas: &tas
            timerange: '2000/2001'
            exp: '2000'
            mip: Amon
            additional_datasets:
              - {project: CMIP6, dataset: B, exp: historical}
              - {project: CMIP6, dataset: A, exp: historical}
          pr:
            <<: *tas
            short_name: pr
        scripts: null
      diag0:
        description: &description Example.
        scripts: null
    preprocessors:
      pp:
        regrid: &regrid {target_grid: 1x1, scheme: linear}
        area_statistics: {operator: mean}
      other:
        regrid: *regrid
    documentation:
      authors: [andela_bouwe]
      description: *description
      title: Example
""")


def test_format_recipe() -> None:
    """Test that sections and facets are put in the standard order."""
    expected = textwrap.dedent("""\
        documentation:
          title: Example
          description: &description Example.
          authors:
            - andela_bouwe

        preprocessors:
          pp:
            regrid: &regrid
              target_grid: 1x1
              scheme: linear
            area_statistics:
              operator: mean
          other:
            regrid: *regrid

        diagnostics:
          diag1:
            variables:
              tas: &tas
                exp: '2000'
                mip: Amon
                timerange: 2000/2001
                additional_datasets:
                  - {dataset: B, exp: historical, project: CMIP6}
                  - {dataset: A, exp: historical, project: CMIP6}
              pr:
                <<: *tas
                short_name: pr
            scripts: null
          diag0:
            description: *description
            scripts: null
    """)
    assert format_recipe(RECIPE) == expected


def test_format_recipe_keeps_data() -> None:
    """Test that formatting does not change the recipe."""
    formatted = format_recipe(RECIPE)
    assert yaml.safe_load(formatted) == yaml.safe_load(RECIPE)
    assert format_recipe(formatted) == formatted


def test_format_recipe_merge_list() -> None:
    """Test that lists of merged mappings are kept on one line."""
    text = textwrap.dedent("""\
        a: &a {x: 1}
        b: &b {y: 2}
        c:
          <<: [*a, *b]
          z: 3
    """)
    assert "  <<: [*a, *b]\n" in format_recipe(text)


def test_format_recipe_self_reference() -> None:
    """Test that a mapping that refers to itself can be formatted."""
    text = "a: &a\n  b: [*a]\n"
    assert format_recipe(text) == "a: &a\n  b:\n    - *a\n"


def test_format_recipe_multiline_string() -> None:
    """Test that strings with multiple lines are written as a block."""
    text = "a:\n  b: 'line 1\n\n    line 2'\n"
    assert format_recipe(text) == "a:\n  b: |-\n    line 1\n    line 2\n"


def test_format_recipe_strips_description() -> None:
    """Test that trailing whitespace is removed from the description."""
    text = "documentation:\n  description: |\n    Line 1. \n    Line 2.\n\n"
    expected = "documentation:\n  description: |-\n    Line 1.\n    Line 2.\n"
    assert format_recipe(text) == expected


def test_format_recipe_empty() -> None:
    """Test that an empty recipe can be formatted."""
    assert format_recipe("") == "{}\n"


def test_format_recipe_invalid() -> None:
    """Test that invalid YAML is reported."""
    with pytest.raises(yaml.YAMLError):
        format_recipe("a: [")


def test_to_yaml_multiline_string() -> None:
    """Test that strings with multiple lines are written as a block."""
    recipe = {"diagnostics": {"d": {"description": "line 1\nline 2\n"}}}
    assert to_yaml(recipe) == (
        "diagnostics:\n  d:\n    description: |\n      line 1\n      line 2"
    )


def test_to_yaml_sorts_numbers() -> None:
    """Test that numbers are sorted before names and by their value."""
    recipe = {"settings": {"b": 0, 10: 0, 2: 0, "a": 0}}
    assert to_yaml(recipe) == "settings:\n  2: 0\n  10: 0\n  a: 0\n  b: 0"
