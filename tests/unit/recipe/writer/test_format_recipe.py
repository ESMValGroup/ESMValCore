"""Unit tests for :func:`esmvalcore.recipe.format_recipe`."""

from __future__ import annotations

import textwrap

import pytest
import yaml

from esmvalcore.recipe import format_recipe

RECIPE = textwrap.dedent("""
    # A comment before the first item stays at the top.
    # Quotes are only kept where they are needed.
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
        # A comment before the first item stays at the top.
        # Quotes are only kept where they are needed.
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


def test_format_recipe_datasets_anchor() -> None:
    """Test that datasets defined at the top level are written on one line."""
    text = textwrap.dedent("""\
        cmip6: &cmip6
          - &defaults
            project: CMIP6
            dataset: A
          - <<: *defaults
            dataset: B
        diagnostics:
          d:
            additional_datasets: *cmip6
    """)
    expected = textwrap.dedent("""\
        cmip6: &cmip6
          - &defaults {dataset: A, project: CMIP6}
          - {<<: *defaults, dataset: B}
    """)
    assert format_recipe(text).startswith(expected)


def test_format_recipe_short_name_first() -> None:
    """Test that short_name is the first key of a variable."""
    text = textwrap.dedent("""\
        diagnostics:
          diag:
            variables:
              tas: &tas
                mip: Amon
                short_name: tas
              pr:
                short_name: pr
                <<: *tas
    """)
    expected = textwrap.dedent("""\
        diagnostics:
          diag:
            variables:
              tas: &tas
                short_name: tas
                mip: Amon
              pr:
                <<: *tas
                short_name: pr
    """)
    assert format_recipe(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("a: [1, 2.5, 3, 4]\n", "a: [1, 2.5, 3, 4]\n"),
        ("a:\n  - 1\n  - 2\n  - 3\n  - 4\n", "a: [1, 2, 3, 4]\n"),
        ("a: [1, 2, 3]\n", "a:\n  - 1\n  - 2\n  - 3\n"),
        ("a: [1, 2, 3, x]\n", "a:\n  - 1\n  - 2\n  - 3\n  - x\n"),
    ],
)
def test_format_recipe_number_list(text: str, expected: str) -> None:
    """Test that lists of more than three numbers are written on one line."""
    assert format_recipe(text) == expected


def test_format_recipe_self_reference() -> None:
    """Test that a mapping that refers to itself can be formatted."""
    text = "a: &a\n  b: [*a]\n"
    assert format_recipe(text) == "a: &a\n  b:\n    - *a\n"


def test_format_recipe_multiline_string() -> None:
    """Test that strings with multiple lines are written as a block."""
    text = "a:\n  b: 'line 1\n\n    line 2'\n"
    assert format_recipe(text) == "a:\n  b: |-\n    line 1\n    line 2\n"


def test_format_recipe_empty() -> None:
    """Test that an empty recipe can be formatted."""
    assert format_recipe("") == "{}\n"


@pytest.mark.parametrize(
    "header",
    [
        "---\n",
        "--- \n",
        "# ESMValTool\n# recipe.yml\n---\n",
        "\n  # Indented\n\n---\n",
        "# Windows\r\n---\r\n",
        "--- # comment\n",
    ],
)
def test_format_recipe_keeps_header(header: str) -> None:
    """Test that the document start and the comments before it are kept."""
    body = "a: 1  # comment\n\n# comment\nb: 2\n"
    formatted = format_recipe(f"{header}{body}")
    assert formatted == f"{header}{body}"
    assert format_recipe(formatted) == formatted


def test_format_recipe_keeps_comments_without_document_start() -> None:
    """Test that comments are kept if there is no document start."""
    assert format_recipe("# comment\na: 1\n") == "# comment\na: 1\n"


@pytest.mark.parametrize("body", ["", "# comment\n"])
def test_format_recipe_only_header(body: str) -> None:
    """Test that a recipe with only a header can be formatted."""
    text = f"# comment\n---\n{body}"
    assert format_recipe(text) == f"# comment\n---\n{body}{{}}\n"


def test_format_recipe_invalid_line_number() -> None:
    """Test that errors are reported at their line in the original text."""
    with pytest.raises(yaml.YAMLError, match="line 4"):
        format_recipe("# comment\n---\na: [\n")


def test_format_recipe_invalid() -> None:
    """Test that invalid YAML is reported."""
    with pytest.raises(yaml.YAMLError):
        format_recipe("a: [")


def test_format_recipe_dict_multiline_string() -> None:
    """Test that strings with multiple lines are written as a block."""
    recipe = {"diagnostics": {"d": {"description": "line 1\nline 2\n"}}}
    assert format_recipe(recipe) == (
        "diagnostics:\n  d:\n    description: |\n      line 1\n      line 2\n"
    )


def test_format_recipe_dict_sorts_numbers() -> None:
    """Test that numbers are sorted before names and by their value."""
    recipe = {"settings": {"b": 0, 10: 0, 2: 0, "a": 0}}
    assert format_recipe(recipe) == (
        "settings:\n  2: 0\n  10: 0\n  a: 0\n  b: 0\n"
    )


def test_format_recipe_description_unchanged() -> None:
    """Test that the text of the description is not changed."""
    text = "documentation:\n  description: |\n    One\n    sentence.\n"
    assert format_recipe(text) == text


def test_description_short_stays_plain() -> None:
    """Test that a short description without line breaks is not a block."""
    text = "documentation:\n  description: Short.\n"
    assert format_recipe(text) == text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param(
            "a:  # one\n  b: 1  # two\n",
            "a:  # one\n  b: 1  # two\n",
            id="trailing",
        ),
        pytest.param(
            "documentation:\n  authors: [a]\n  # The title.\n  title: T\n",
            "documentation:\n  # The title.\n  title: T\n  authors:\n"
            "    - a\n",
            id="moves-with-item",
        ),
        pytest.param(
            "datasets:\n  - {dataset: A}\n  # - {dataset: B}\n"
            "  - {dataset: C}\n  # - {dataset: D}\n",
            "datasets:\n  - {dataset: A}\n  # - {dataset: B}\n"
            "  - {dataset: C}\n  # - {dataset: D}\n",
            id="commented-out-datasets",
        ),
        pytest.param(
            "a:\n  b:\n    c: 1\n    # d: 2\n  e: 3\n",
            "a:\n  b:\n    c: 1\n    # d: 2\n  e: 3\n",
            id="end-of-block",
        ),
        pytest.param(
            "datasets:\n  - dataset: A  # one\n    # exp: x\n"
            "    project: P  # two\n",
            "datasets:\n  # exp: x\n  # two\n  - {dataset: A, project: P}  # one\n",
            id="in-dataset",
        ),
        pytest.param(
            "datasets:\n  - {dataset: A,  # one\n     exp: x}\n",
            "datasets:\n  - {dataset: A, exp: x}  # one\n",
            id="in-flow-mapping",
        ),
        pytest.param(
            "datasets:\n  - {dataset: A, supplementary_variables: [\n"
            "      {short_name: sftlf}]}  # one\n",
            "datasets:\n  # one\n  - {dataset: A, supplementary_variables: "
            "[{short_name: sftlf}]}\n",
            id="in-nested-flow",
        ),
        pytest.param(
            "a: [1,  # one\n  2]\n",
            "a:\n  - 1  # one\n  - 2\n",
            id="flow-list-becomes-block",
        ),
        pytest.param(
            "a: |  # one\n  x\n  y\n",
            "a: |  # one\n  x\n  y\n",
            id="block-scalar-header",
        ),
        pytest.param(
            "# schema\n\na: 1\n",
            "# schema\n\na: 1\n",
            id="blank-line-after-leading",
        ),
        pytest.param(
            "a:\n  # one\n\n  # two\n  b: 1\n",
            "a:\n  # one\n\n  # two\n  b: 1\n",
            id="blank-line-between-comments",
        ),
        pytest.param(
            "documentation:\n  title: T\n# Datasets\ndatasets: []\n",
            "documentation:\n  title: T\n\n# Datasets\ndatasets: []\n",
            id="top-level-blank-line",
        ),
        pytest.param(
            "a: &a  # one\n  x: 1\nb:\n  <<: *a  # two\n  y: 2\n"
            "c:\n  - &c 1\n  # three\n  - *c\n",
            "a: &a  # one\n  x: 1\n\nb:\n  <<: *a  # two\n  y: 2\n\n"
            "c:\n  - &c 1\n  # three\n  - *c\n",
            id="anchors-and-aliases",
        ),
        pytest.param(
            "a: 'x # y'\nb: |\n  # not a comment\n",
            "a: 'x # y'\n\nb: |\n  # not a comment\n",
            id="not-comments",
        ),
        pytest.param(
            "a: 1\n# end\n",
            "a: 1\n# end\n",
            id="end-of-document",
        ),
    ],
)
def test_format_recipe_keeps_comments(text: str, expected: str) -> None:
    """Test that comments are kept with the item they belong to."""
    formatted = format_recipe(text)
    assert formatted == expected
    assert yaml.safe_load(formatted) == yaml.safe_load(text)
    assert format_recipe(formatted) == formatted


def test_format_recipe_comment_after_wrapped_scalar() -> None:
    """Test that a comment is written after a scalar that is wrapped."""
    words = " ".join(["word"] * 60)
    text = f"a: {words}  # comment\nb: 1\n"
    formatted = format_recipe(text)
    assert yaml.safe_load(formatted) == yaml.safe_load(text)
    lines = formatted.splitlines()
    assert len(lines) > 2
    assert lines[-3].endswith(" word  # comment")


def test_format_recipe_comment_after_description_header() -> None:
    """Test that a comment after the header of the description is kept."""
    text = "documentation:\n  description: |  # one\n    Some\n    text.\n"
    formatted = format_recipe(text)
    assert formatted == text


def test_format_recipe_byte_order_mark() -> None:
    """Test that a byte order mark at the start of a recipe is skipped."""
    assert format_recipe("\ufeffa: 1\n") == "a: 1\n"


@pytest.mark.parametrize(
    "text",
    [
        "datasets: abc\n",
        "diagnostics:\n  d:\n    additional_datasets: abc\n",
    ],
)
def test_format_recipe_datasets_not_a_list(text: str) -> None:
    """Test that invalid lists of datasets are formatted as other values."""
    assert format_recipe(text) == text


def test_format_recipe_dict_sorts_infinite_number_last() -> None:
    """Test that a key that cannot be converted to a number sorts last."""
    recipe = {"a": {float("inf"): 1, 2: 3}}
    assert format_recipe(recipe) == "a:\n  2: 3\n  .inf: 1\n"


def test_format_recipe_blank_lines_between_top_level_keys() -> None:
    """Test that there is a blank line between all top-level keys."""
    text = "a: 1\nb:\n  - 1\nc: 2\n"
    assert format_recipe(text) == "a: 1\n\nb:\n  - 1\n\nc: 2\n"


def test_format_recipe_keeps_order_of_other_top_level_keys() -> None:
    """Test that top-level keys without a standard order keep their order."""
    text = "diagnostics: {}\nb: 1\na: 2\ndocumentation: {}\n"
    expected = "documentation: {}\n\nb: 1\n\na: 2\n\ndiagnostics: {}\n"
    assert format_recipe(text) == expected


def test_format_recipe_comment_with_dash_after_item_dash() -> None:
    """Test a comment with a "-" between the "-" of an item and its value."""
    text = "a:\n  - # note-x\n    b: 1\n  - c\n  # after c\nd: 1\n"
    expected = "a:\n  - b: 1  # note-x\n  - c\n  # after c\n\nd: 1\n"
    assert format_recipe(text) == expected
