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
    text = "documentation:\n  description: |\n    Line 1. \n\n    Line 2.\n\n"
    expected = (
        "documentation:\n  description: |-\n    Line 1.\n\n    Line 2.\n"
    )
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


def _description(text: str) -> str:
    """Load the description of a formatted recipe."""
    return yaml.safe_load(text)["documentation"]["description"]


DESCRIPTION = (
    "documentation:\n  description: |\n"
    "    This sentence is split\n    over two lines.\n\n"
    "    Second paragraph, which is\n    also split.\n"
    "    - item one\n      continues here\n"
    "    - item two\n"
    "        - nested item\n"
    "    Text after the list.\n"
)
NORMALIZED = (
    "This sentence is split over two lines.\n\n"
    "Second paragraph, which is also split.\n"
    "- item one continues here\n"
    "- item two\n"
    "    - nested item\n"
    "Text after the list."
)


def test_format_recipe_description_joins_lines() -> None:
    """Test that the lines of a paragraph are joined."""
    text = "documentation:\n  description: |\n    One\n    sentence.\n"
    assert _description(format_recipe(text)) == "One sentence."


def test_format_recipe_description_keeps_lists_and_paragraphs() -> None:
    """Test that list items, indented lines and paragraphs are kept."""
    formatted = format_recipe(DESCRIPTION)
    assert _description(formatted) == NORMALIZED
    assert "  description: |-\n" in formatted


FOLDED = """\
    First paragraph,
    split over two lines.

    Second paragraph.

    Notes:

    - item one
    - item two
"""
FOLDED_NORMALIZED = (
    "First paragraph, split over two lines.\n\n"
    "Second paragraph.\n\n"
    "Notes:\n\n"
    "- item one\n"
    "- item two"
)


@pytest.mark.parametrize(
    "header",
    [">", ">-", ">+", "> # comment", "&anchor >", "!!str >", "&anchor\n    >"],
)
def test_format_recipe_folded_description(header: str) -> None:
    """Test that the paragraphs and lists of a folded block are kept."""
    text = (
        f"documentation:\n  description: {header}\n{FOLDED}  title: Example\n"
    )
    formatted = format_recipe(text)
    assert _description(formatted) == FOLDED_NORMALIZED
    assert yaml.safe_load(formatted)["documentation"]["title"] == "Example"
    assert format_recipe(formatted) == formatted


def test_format_recipe_folded_description_at_end() -> None:
    """Test a folded block at the end of a recipe without a final newline."""
    text = "documentation:\n  description: >\n    One\n    sentence."
    assert _description(format_recipe(text)) == "One sentence."


WORDS = " ".join(["word"] * 30)
LONG = f"{WORDS}\n\n{WORDS}\n  indented {WORDS}\n- item {WORDS}\n1. {WORDS}"


def _wrap(text: str, indent: str = "") -> str:
    """Wrap text at 75 characters."""
    return textwrap.fill(text, 75, subsequent_indent=indent)


@pytest.mark.parametrize("use_to_yaml", [True, False])
def test_description_idempotent_and_wrapped(*, use_to_yaml: bool) -> None:
    """Test that formatting twice gives the same text, wrapped at 79."""
    if use_to_yaml:
        formatted = to_yaml({"documentation": {"description": LONG}})
        again = to_yaml(yaml.safe_load(formatted))
    else:
        indented = textwrap.indent(LONG, "    ")
        formatted = format_recipe(
            f"documentation:\n  description: |\n{indented}\n",
        )
        again = format_recipe(formatted)
    assert again == formatted
    lines = formatted.splitlines()
    assert lines[1] == "  description: |-"
    assert max(len(line) for line in lines if "indented" not in line) <= 79
    description = _description(formatted)
    assert description == "\n".join(
        [
            _wrap(WORDS),
            "",
            _wrap(WORDS),
            f"  indented {WORDS}",
            _wrap(f"- item {WORDS}", "  "),
            _wrap(f"1. {WORDS}", "   "),
        ],
    )
    assert "\n  indented word word" in description


def test_description_list_item_is_wrapped_with_hanging_indent() -> None:
    """Test that continuation lines of a list item are indented."""
    recipe = {"documentation": {"description": f"- {WORDS}"}}
    lines = _description(to_yaml(recipe)).splitlines()
    assert len(lines) > 1
    assert lines[1].startswith("  word")


def test_description_wrapping_does_not_create_list_items() -> None:
    """Test that a wrapped line does not start with a list marker."""
    text = " ".join(["word"] * 18) + " - " + "word " * 5
    recipe = {"documentation": {"description": text.strip()}}
    formatted = to_yaml(recipe)
    assert to_yaml(yaml.safe_load(formatted)) == formatted
    assert not any(
        line.lstrip().startswith("- ") for line in formatted.splitlines()[2:]
    )


def test_to_yaml_description_normalized() -> None:
    """Test that to_yaml normalizes the description."""
    recipe = {"documentation": {"description": "a\nb\n\n  - c\n    d\n"}}
    assert _description(to_yaml(recipe)) == "a b\n\n  - c d"


def test_description_short_stays_plain() -> None:
    """Test that a short description without line breaks is not a block."""
    text = "documentation:\n  description: Short.\n"
    assert format_recipe(text) == text
