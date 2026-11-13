"""Tests for :mod:`esmvalcore._recipe.validate`."""

from __future__ import annotations

import copy
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import jsonschema
import pytest
import yaml

import esmvalcore.io.esgf.facets
import esmvalcore.preprocessor
from esmvalcore._recipe import validate
from esmvalcore._recipe.validate import (
    RecipeIssue,
    ensure_valid_recipe,
    validate_recipe,
)
from esmvalcore.config import CFG
from esmvalcore.config._diagnostics import Diagnostics
from esmvalcore.exceptions import RecipeError
from tests.unit.recipe.example_recipes import (
    AUTHORS,
    DUPLICATE_RECIPE,
    INVALID_RECIPE,
    RECIPE,
)

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

    from esmvalcore.config import Session


def _issues(
    recipe: object,
    session: Session | None = None,
) -> list[RecipeIssue]:
    """Validate a recipe that is already loaded, so without positions."""
    return validate._validate(recipe, None, session, required_facets=True)


SMALL_RECIPE: dict[str, Any] = {
    "documentation": {
        "title": "Example",
        "description": "Example recipe.",
        "authors": ["andela_bouwe"],
    },
    "datasets": [
        {
            "project": "CMIP6",
            "dataset": "EC-Earth3",
            "exp": "historical",
            "ensemble": "r1i1p1f1",
            "grid": "gr",
        },
    ],
    "preprocessors": {"pp": {}},
    "diagnostics": {
        "example": {
            "variables": {"tas": {"mip": "Amon"}},
            "scripts": None,
        },
    },
}


def _messages(preprocessor: dict[str, Any]) -> list[str]:
    recipe = {**SMALL_RECIPE, "preprocessors": {"pp": preprocessor}}
    return [str(issue) for issue in _issues(recipe)]


def test_diagnostics_null() -> None:
    recipe = {**SMALL_RECIPE, "diagnostics": None}
    assert [str(i) for i in _issues(recipe)] == [
        "diagnostics: expected a mapping, got None.",
    ]


def test_script_named_like_variable_group() -> None:
    recipe = {
        **SMALL_RECIPE,
        "diagnostics": {
            "example": {
                "variables": {"tas": {"mip": "Amon"}},
                "scripts": {"tas": {"script": "example.py"}},
            },
        },
    }
    message = (
        "diagnostics > example > scripts > tas: a script cannot have the "
        "same name as a variable group of its diagnostic."
    )
    assert [str(i) for i in _issues(recipe)] == [message]


@pytest.mark.parametrize(
    "settings",
    [
        {"operator": "mean"},
        {"operator": "percentile", "percent": 50},
        {"hours": 6, "operator": "max"},
    ],
)
def test_statistics_valid(settings: dict[str, Any]) -> None:
    step = "hourly_statistics" if "hours" in settings else "annual_statistics"
    assert _messages({step: settings}) == []


def test_statistics_invalid_kwargs() -> None:
    (message,) = _messages(
        {"annual_statistics": {"operator": "mean", "percent": 50}},
    )
    assert message.startswith(
        "preprocessors > pp > annual_statistics: Invalid kwargs for operator "
        "'mean': ",
    )


@pytest.mark.parametrize(
    "step",
    ["multi_model_statistics", "ensemble_statistics"],
)
def test_multi_model_statistics_invalid_kwargs(step: str) -> None:
    settings = {
        "span": "overlap",
        "statistics": ["mean", {"operator": "mean", "percent": 50}],
    }
    (message,) = _messages({step: settings})
    assert message.startswith(
        f"preprocessors > pp > {step} > statistics > 1: Invalid kwargs for "
        "operator 'mean': ",
    )


def test_generic_regridding_scheme_valid() -> None:
    scheme = {"reference": "iris.analysis:Linear"}
    regrid = {"target_grid": "1x1", "scheme": scheme}
    assert _messages({"regrid": regrid}) == []


def test_generic_regridding_scheme_invalid() -> None:
    scheme = {"reference": "invalid.module:function"}
    (message,) = _messages(
        {"regrid": {"target_grid": "1x1", "scheme": scheme}},
    )
    assert message.startswith(
        "preprocessors > pp > regrid > scheme: failed to load generic "
        "regridding scheme: Could not import specified generic regridding "
        "module 'invalid.module'.",
    )


@pytest.mark.parametrize(
    ("scheme", "error"),
    [
        ({"reference": "iris.analysis:Invalid"}, "has no attribute 'Invalid'"),
        ({"reference": "iris.analysis:Linear", "invalid": 1}, "unexpected"),
    ],
)
def test_generic_regridding_scheme_cannot_load(
    scheme: dict[str, Any],
    error: str,
) -> None:
    (message,) = _messages(
        {"regrid": {"target_grid": "1x1", "scheme": scheme}},
    )
    assert message.startswith(
        "preprocessors > pp > regrid > scheme: failed to load generic "
        "regridding scheme: ",
    )
    assert error in message


@pytest.mark.parametrize(
    ("scheme", "message"),
    [
        ({}, "scheme: missing required key 'reference'."),
        ({"reference": 1}, "scheme > reference: expected a string, got 1."),
    ],
)
def test_generic_regridding_scheme_invalid_reference(
    scheme: dict[str, Any],
    message: str,
) -> None:
    assert _messages({"regrid": {"target_grid": "1x1", "scheme": scheme}}) == [
        f"preprocessors > pp > regrid > {message}",
    ]


def test_statistics_kwarg_not_a_string() -> None:
    assert _messages({"annual_statistics": {"operator": "mean", 1: 2}}) == [
        "preprocessors > pp > annual_statistics: keywords must be strings.",
    ]


def test_unknown_key_not_a_string() -> None:
    recipe = yaml.safe_load(RECIPE)
    recipe["documentation"][2024] = "text"
    issues = [i for i in _issues(recipe) if i.path[-1] == 2024]
    assert [i.message.split(".")[0] for i in issues] == ["unknown key"]


def test_failing_check_is_a_warning(mocker: MockerFixture) -> None:
    mocker.patch.object(
        validate,
        "_find_unknown_types",
        autospec=True,
        side_effect=KeyError("bug"),
    )
    recipe = yaml.safe_load(RECIPE)
    recipe["diagnostics"]["example"]["variables"]["ta"]["preprocessor"] = "x"
    issues = _issues(recipe)
    assert (
        RecipeIssue(
            (),
            "failed to run check '_find_unknown_types': KeyError('bug'). This is "
            "a bug, please report it at "
            "https://github.com/ESMValGroup/ESMValCore/issues.",
            severity="warning",
        )
        in issues
    )
    # The other checks still run.
    assert any(
        "Not defined in the `preprocessors`" in i.message for i in issues
    )


def test_session_configuration(session: Session) -> None:
    """Test that the configuration of the session is used if given."""
    projects = dict(session["projects"])
    projects["CMIP6"] = {**projects["CMIP6"], "required_facets": ["institute"]}
    session["projects"] = projects
    dataset = {
        "project": "CMIP6",
        "dataset": "EC-Earth3",
        "exp": "historical",
        "ensemble": "r1i1p1f1",
        "grid": "gr",
    }
    recipe = {**SMALL_RECIPE, "datasets": [dataset]}
    assert _issues(recipe) == []
    message = (
        "datasets > 0: missing required key 'institute'. Needed to find the "
        "data of project 'CMIP6'."
    )
    issues = _issues(recipe, session=session)
    assert [str(i) for i in issues] == [message]


@pytest.mark.parametrize(
    ("variable", "datasets", "path", "message"),
    [
        (
            {"mip": "Amon"},
            [],
            ("diagnostics", "example", "variables", "tas"),
            (
                "no datasets. Add them to the `datasets` section or to "
                "`additional_datasets`."
            ),
        ),
        (
            {"mip": "Amon"},
            [{"dataset": "A"}, {"dataset": "A"}],
            ("datasets", 1),
            "duplicate dataset.",
        ),
        (
            {},
            [{"dataset": "A"}],
            ("datasets", 0),
            "missing required key 'mip'.",
        ),
    ],
)
def test_incomplete_datasets(
    variable: dict[str, Any],
    datasets: list[dict[str, Any]],
    path: tuple[str | int, ...],
    message: str,
) -> None:
    recipe = {
        **SMALL_RECIPE,
        "datasets": [{"project": "OBS", "tier": 2, **d} for d in datasets],
        "diagnostics": {
            "example": {"variables": {"tas": variable}, "scripts": None},
        },
    }
    issues = [i for i in _issues(recipe) if "Needed to find" not in i.message]
    assert [(i.path, i.message) for i in issues] == [(path, message)]


def test_missing_project() -> None:
    recipe = {
        **SMALL_RECIPE,
        "datasets": [{"dataset": "A"}],
    }
    assert [str(i) for i in _issues(recipe)] == [
        "datasets > 0: missing required key 'project'.",
    ]


def test_null_variable_group() -> None:
    """Test a variable group without facets, which its datasets define."""
    text = """
documentation: {title: Test, description: Test., authors: [andela_bouwe]}
diagnostics:
  example:
    variables:
      pr:
    additional_datasets:
      - {dataset: A, project: OBS, type: sat, tier: 2, version: 1, mip: Amon}
    scripts: null
"""
    assert validate_recipe(text) == []


def test_null_variable_group_without_mip() -> None:
    text = """
documentation: {title: Test, description: Test., authors: [andela_bouwe]}
diagnostics:
  example:
    variables:
      pr:
    additional_datasets:
      - {dataset: A, project: OBS, type: sat, tier: 2, version: 1}
    scripts: null
"""
    message = (
        "line 8: diagnostics > example > additional_datasets > 0: missing "
        "required key 'mip'."
    )
    assert [str(i) for i in validate_recipe(text)] == [message]


def test_validate_recipe_valid() -> None:
    assert _issues(yaml.safe_load(RECIPE)) == []


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (
            ["preprocessors", "pp", "regrid"],
            {"scheme": "linear"},
            (
                "preprocessors > pp > regrid: missing required argument "
                "'target_grid'."
            ),
        ),
        (
            ["preprocessors", "pp", "regrid"],
            None,
            (
                "preprocessors > pp > regrid: missing required arguments "
                "'target_grid', 'scheme'."
            ),
        ),
        (
            ["preprocessors", "pp", "regrid"],
            True,
            (
                "preprocessors > pp > regrid: missing required arguments "
                "'target_grid', 'scheme'."
            ),
        ),
        (
            ["preprocessors", "pp", "regrid", "target_scheme"],
            "linear",
            (
                "preprocessors > pp > regrid > target_scheme: unknown argument. "
                "Did you mean 'scheme'? Valid arguments are: cache_weights, "
                "lat_offset, lon_offset, scheme, target_grid, use_src_coords."
            ),
        ),
        (
            ["preprocessors", "pp", "regird"],
            {},
            (
                "preprocessors > pp > regird: unknown preprocessor function. "
                "Did you mean 'regrid'?"
            ),
        ),
        (
            ["preprocessors", "pp", "extract_levels", "scheme"],
            "cubic",
            (
                "preprocessors > pp > extract_levels > scheme: invalid value "
                "'cubic', choose from 'linear', 'nearest', 'linear_extrapolate', "
                "'nearest_extrapolate'."
            ),
        ),
        (
            ["preprocessors", "pp", "regrid", "scheme"],
            "cubic",
            (
                "preprocessors > pp > regrid > scheme: invalid value 'cubic', "
                "choose from 'area_weighted', 'linear', 'nearest', or a "
                "mapping."
            ),
        ),
        (
            ["preprocessors", "pp", "regrid", "target_grid"],
            1,
            (
                "preprocessors > pp > regrid > target_grid: expected a string or "
                "a mapping, got 1."
            ),
        ),
        (
            ["preprocessors", "pp", "extract_levels", "coordinate"],
            1,
            (
                "preprocessors > pp > extract_levels > coordinate: expected a "
                "string or nothing, got 1."
            ),
        ),
        (
            ["datasets", 0, "dataset"],
            None,
            "datasets > 0 > dataset: expected a string, got None.",
        ),
        (
            ["datasets", 0, "grid"],
            {"label": "gn"},
            (
                "datasets > 0 > grid: expected a string, a number, or a list, got "
                "{'label': 'gn'}."
            ),
        ),
        (
            ["documentation", "authors"],
            [],
            "documentation > authors: expected at least 1 item, got [].",
        ),
        (
            ["preprocessors", "pp", "histogram"],
            {"bin_range": [0, 1, 2]},
            (
                "preprocessors > pp > histogram > bin_range: expected at most "
                "2 items, got [0, 1, 2]."
            ),
        ),
        (
            ["preprocessors", "pp"],
            None,
            "preprocessors > pp: expected a mapping, got None.",
        ),
        (
            ["diagnostics", "example", "scripts", "plot", "ancestors"],
            [1],
            (
                "diagnostics > example > scripts > plot > ancestors > 0: "
                "expected a string, got 1."
            ),
        ),
        (
            ["diagnostics", "example", "scripts", "plot", "script"],
            1,
            (
                "diagnostics > example > scripts > plot > script: expected a "
                "string, got 1."
            ),
        ),
        (
            ["datasets", 1, "tier"],
            7,
            "datasets > 1 > tier: expected at most 3, got 7.",
        ),
    ],
)
def test_validate_recipe_invalid(
    path: list[str | int],
    value: object,
    message: str,
) -> None:
    recipe = yaml.safe_load(RECIPE)
    parent = recipe
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    assert [str(i) for i in _issues(recipe)] == [message]


@pytest.mark.parametrize(
    ("path", "message"),
    [
        (
            ["documentation", "project"],
            "unknown key. Did you mean 'projects'?",
        ),
        (["diagnostics", "example", "plot_types"], "unknown key."),
    ],
)
def test_validate_recipe_unknown_key_deprecated(
    path: list[str],
    message: str,
) -> None:
    recipe = yaml.safe_load(RECIPE)
    parent = recipe
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = ["value"]
    message += (
        " Unknown keys are ignored, this is deprecated and will not work in "
        "v2.18.0."
    )
    assert _issues(recipe) == [
        RecipeIssue(tuple(path), message, severity="warning"),
    ]


def test_describe_ambiguous_type() -> None:
    # Both options accept any type, so the error cannot be described in
    # terms of the expected types.
    schema = {
        "definitions": {"a": {"type": "string"}, "b": {"type": "object"}},
        "anyOf": [{"$ref": "#/definitions/a"}, {"$ref": "#/definitions/b"}],
    }
    error = next(jsonschema.Draft7Validator(schema).iter_errors(1))
    assert validate._describe(error, []) == [
        RecipeIssue((), error.message),
    ]


def test_validate_recipe_not_a_mapping() -> None:
    assert _issues(None) == [
        RecipeIssue((), "expected a mapping, got None."),
    ]


def test_validate_recipe_multiple_problems() -> None:
    recipe = yaml.safe_load(RECIPE)
    recipe["preprocessors"]["pp"]["regird"] = {}
    recipe["diagnostics"]["example"].pop("scripts")
    assert [str(i) for i in _issues(recipe)] == [
        (
            "preprocessors > pp > regird: unknown preprocessor function. "
            "Did you mean 'regrid'?"
        ),
        "diagnostics > example: missing required key 'scripts'.",
    ]


def test_validate_recipe_positions() -> None:
    issues = validate_recipe(INVALID_RECIPE)
    assert issues == [
        RecipeIssue(
            ("datasets", 1),
            "missing required key 'dataset'.",
            line=5,
            column=5,
        ),
        RecipeIssue(
            ("datasets", 1, "tier"),
            "expected at most 3, got 7.",
            line=5,
            column=22,
        ),
        # Problems in an anchored mapping are found at the anchor, and where
        # an alias uses it.
        RecipeIssue(
            ("preprocessors", "base", "area_statistics", "operator"),
            "expected a string, got 1.",
            line=9,
            column=7,
        ),
        RecipeIssue(
            ("preprocessors", "derived", "area_statistics", "operator"),
            "expected a string, got 1.",
            line=11,
            column=9,
        ),
        RecipeIssue(
            ("preprocessors", "derived", "convert_units"),
            "missing required argument 'units'.",
            line=12,
            column=5,
        ),
        RecipeIssue(
            ("diagnostics", "example"),
            "missing required key 'scripts'.",
            line=14,
            column=3,
        ),
    ]
    assert str(issues[0]) == (
        "line 5: datasets > 1: missing required key 'dataset'."
    )


MERGE_RECIPE = """
a: &a
  scheme: linear
b: &b
  scheme: nearest
  target_grid: 1x1
preprocessors:
  sequence:
    regrid:
      <<: [*a, *b]
  repeated:
    regrid:
      <<: *a
      <<: *b
"""


@pytest.mark.parametrize(
    ("preprocessor", "scheme", "line", "column"),
    [
        # In a sequence of mappings to merge, the first one has priority.
        ("sequence", "linear", 10, 12),
        # With repeated merge keys, the last one has priority.
        ("repeated", "nearest", 14, 11),
    ],
)
def test_validate_recipe_positions_merge(
    preprocessor: str,
    scheme: str,
    line: int,
    column: int,
) -> None:
    recipe = yaml.safe_load(MERGE_RECIPE)
    assert recipe["preprocessors"][preprocessor]["regrid"]["scheme"] == scheme
    recipe["preprocessors"][preprocessor]["regrid"]["scheme"] = "invalid"
    _, composed = validate._load(MERGE_RECIPE)
    issues = [
        issue
        for issue in validate._validate(
            recipe,
            composed,
            None,
            required_facets=True,
        )
        if issue.path[:2] == ("preprocessors", preprocessor)
        and issue.severity == "error"
    ]
    assert len(issues) == 1
    assert issues[0].path[-1] == "scheme"
    # The problem is found at the alias that merges the anchored mapping.
    assert (issues[0].line, issues[0].column) == (line, column)


def test_validate_recipe_duplicate_merge_keys() -> None:
    issues = [
        issue
        for issue in validate_recipe(MERGE_RECIPE)
        if "merge key" in issue.message
    ]
    assert issues == [
        RecipeIssue(
            ("preprocessors", "repeated", "regrid"),
            "duplicate merge key '<<', which other YAML parsers do not "
            "accept. This is deprecated and will not work in v2.18.0: use "
            "one merge key with the mappings in this order: <<: [*b, *a]",
            line=14,
            column=7,
            severity="warning",
        ),
    ]


def test_validate_recipe_duplicate_merge_keys_sequence() -> None:
    text = MERGE_RECIPE.replace("<<: *b", "<<: [*b, {method: x}]")
    messages = [
        issue.message
        for issue in validate_recipe(text)
        if "merge key" in issue.message
    ]
    assert messages[0].endswith("<<: [*b, {...}, *a]")


SELF_MERGE_RECIPE = """
defaults: &defaults {mip: Amon}
diagnostics:
  example:
    variables:
      tas: &tas
        <<: [*defaults, *tas]
        short_name: tas
    scripts: null
"""


def test_validate_recipe_self_merge() -> None:
    issues = validate_recipe(SELF_MERGE_RECIPE)
    assert [i for i in issues if "alias" in i.message] == [
        RecipeIssue(
            ("diagnostics", "example", "variables", "tas"),
            "alias '*tas' refers to the anchor '&tas' that contains it, "
            "which other YAML parsers do not accept. This is deprecated and "
            "will not work in v2.18.0: remove the alias.",
            line=7,
            column=25,
            severity="warning",
        ),
    ]


@pytest.mark.parametrize(
    ("value", "read", "hint"),
    [
        (
            "1e-2",
            "as text by ESMValCore but as a number",
            "quote it if it is text, or write 0.01 for the number",
        ),
        (
            "0o12",
            "as text by ESMValCore but as an integer",
            "quote it if it is text, or write 10 for the integer",
        ),
        (
            "yes",
            "as a boolean by ESMValCore but as text",
            "write true for a boolean, or quote it if it is text",
        ),
        (
            "012",
            "as 10 by ESMValCore but as 12",
            "write 10 for an integer, or quote it if it is text",
        ),
        (
            "1_000",
            "as an integer by ESMValCore but as text",
            "write 1000 for an integer, or quote it if it is text",
        ),
        (
            "1:30",
            "as an integer by ESMValCore but as text",
            "write 90 for an integer, or quote it if it is text",
        ),
    ],
)
def test_validate_recipe_ambiguous_value(
    value: str,
    read: str,
    hint: str,
) -> None:
    text = f"{RECIPE}\n        threshold: {value}\n"
    issues = [
        issue
        for issue in validate_recipe(text)
        if "ambiguous" in issue.message
    ]
    assert [(i.path, i.message, i.severity) for i in issues] == [
        (
            ("diagnostics", "example", "scripts", "plot", "threshold"),
            (
                f"ambiguous value '{value}', read {read} by YAML 1.2 "
                "parsers, for example in editors. This is deprecated and "
                f"will not work in v2.18.0: {hint}."
            ),
            "warning",
        ),
    ]


@pytest.mark.parametrize(
    "value",
    [
        "'1e-2'",
        "1.0e-2",
        "1.0e+2",
        "0.01",
        "10",
        "0x1F",
        "true",
        "~",
        "2000-01-01",
        "text",
    ],
)
def test_validate_recipe_unambiguous_value(value: str) -> None:
    text = f"{RECIPE}\n        threshold: {value}\n"
    issues = validate_recipe(text)
    assert not [i for i in issues if "ambiguous" in i.message]


ALIAS_RECIPE = """
datasets:
  - &dataset {dataset: CanESM2, project: CMIP5, tier: 7}
  - *dataset
preprocessors:
  base: &base
    regrid: &regrid
      scheme: invalid
  derived:
    <<: *base
  value:
    regrid: *regrid
"""


def test_validate_recipe_positions_alias() -> None:
    issues = validate_recipe(ALIAS_RECIPE)
    positions = {
        issue.path: (issue.line, issue.column)
        for issue in issues
        if issue.path[-1:] in {("tier",), ("scheme",)}
    }
    # The value of the item at the alias in a sequence, of the item merged
    # in by an alias, and of the item with an alias as value are found at
    # the alias. Only the outermost alias matters.
    assert positions == {
        ("datasets", 0, "tier"): (3, 49),
        ("datasets", 1, "tier"): (4, 5),
        ("preprocessors", "base", "regrid", "scheme"): (8, 7),
        ("preprocessors", "derived", "regrid", "scheme"): (10, 9),
        ("preprocessors", "value", "regrid", "scheme"): (12, 13),
    }


@pytest.mark.parametrize("text", [RECIPE, INVALID_RECIPE, DUPLICATE_RECIPE])
def test_load(text: str) -> None:
    recipe, composed = validate._load(text)
    assert recipe == yaml.safe_load(text)
    assert composed is not None


def test_validate_recipe_from_invalid_text() -> None:
    issues = validate_recipe("documentation:\n  title: [\n")
    assert len(issues) == 1
    assert issues[0].path == ()
    assert issues[0].line == 3
    assert issues[0].message.startswith("invalid YAML: ")


def test_validate_recipe_from_unreadable_text() -> None:
    (issue,) = validate_recipe("documentation: \x07\n")
    assert issue.path == ()
    assert issue.line is None
    assert issue.message.startswith(
        "invalid YAML: unacceptable character #x0007",
    )


def test_validate_recipe_from_empty_text() -> None:
    assert validate_recipe("") == _issues(None)


def test_validate_recipe_key_without_position() -> None:
    # The key is read as the number 16, which does not match its text.
    issues = validate_recipe("documentation:\n  0x10: value\n")
    assert issues[-1].path == ("documentation", 16)
    assert issues[-1].line is None


def test_validate_recipe_does_not_modify_recipe() -> None:
    recipe = yaml.safe_load(RECIPE)
    recipe["preprocessors"]["pp"]["regrid"] = None
    _issues(recipe)
    assert recipe["preprocessors"]["pp"]["regrid"] is None


def test_validate_recipe_duplicate_keys() -> None:
    issues = validate_recipe(DUPLICATE_RECIPE)
    assert issues == [
        # Problems in an anchored mapping are only reported once.
        RecipeIssue(
            ("preprocessors", "base", "area_statistics"),
            "duplicate key, this definition is ignored because "
            "'area_statistics' is defined again on line 7.",
            line=5,
            column=5,
            severity="warning",
        ),
        RecipeIssue(
            ("diagnostics", "example", "variables", "tas"),
            "duplicate key, this definition is ignored because 'tas' is "
            "defined again on line 18.",
            line=16,
            column=7,
            severity="warning",
        ),
    ]


def test_validate_recipe_repeated_merge_keys() -> None:
    # MERGE_RECIPE uses the merge key (<<) more than once in a mapping.
    issues = validate_recipe(MERGE_RECIPE)
    assert not [i for i in issues if "duplicate key" in i.message]


def test_validate_recipe_unknown_preprocessor() -> None:
    recipe = yaml.safe_load(RECIPE)
    recipe["diagnostics"]["example"]["variables"]["ta"]["preprocessor"] = "p"
    assert _issues(recipe) == [
        RecipeIssue(
            ("diagnostics", "example", "variables", "ta", "preprocessor"),
            "unknown value 'p'. Not defined in the `preprocessors` section. "
            "Did you mean 'pp'?",
        ),
    ]


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("datasets", 0), "p"),
        (("datasets", 1), None),
        (
            (
                "diagnostics",
                "example",
                "variables",
                "ta",
                "additional_datasets",
                0,
            ),
            "p",
        ),
    ],
)
def test_validate_recipe_unknown_dataset_preprocessor(
    path: tuple[str | int, ...],
    value: str | None,
) -> None:
    recipe = yaml.safe_load(RECIPE)
    node = recipe
    for key in path:
        node = node[key]
    node["preprocessor"] = value
    suggestion = " Did you mean 'pp'?" if value == "p" else ""
    assert _issues(recipe) == [
        RecipeIssue(
            (*path, "preprocessor"),
            f"unknown value {value!r}. Not defined in the `preprocessors` "
            f"section.{suggestion}",
        ),
    ]


def test_validate_recipe_default_preprocessor() -> None:
    recipe = yaml.safe_load(RECIPE)
    del recipe["preprocessors"]["default"]
    variable = recipe["diagnostics"]["example"]["variables"]["ta"]
    variable["preprocessor"] = "default"
    assert _issues(recipe) == []


@pytest.mark.parametrize("project", ["OBS", "OBS6"])
def test_validate_recipe_missing_facets(project: str) -> None:
    recipe = yaml.safe_load(RECIPE)
    # The dataset is used by both variable groups, but reported once.
    recipe["datasets"][1] = {"dataset": "ERA5", "project": project}
    assert _issues(recipe) == [
        RecipeIssue(
            ("datasets", 1),
            "missing required keys 'tier', 'type'. Needed to find the data "
            f"of project '{project}'.",
        ),
    ]


def test_validate_recipe_facets_of_variable_group() -> None:
    recipe = yaml.safe_load(RECIPE)
    variable = recipe["diagnostics"]["example"]["variables"]["ta"]
    variable.update(project="OBS", tier=2)
    variable["additional_datasets"] = [
        {"dataset": "HadISST", "type": "reanaly"},
        {"dataset": "CRU"},
        {
            "dataset": "CanESM5",
            "project": "CMIP6",
            "exp": "historical",
            "ensemble": "r1i1p1f1",
            "grid": "gn",
        },
    ]
    assert _issues(recipe) == [
        RecipeIssue(
            (
                "diagnostics",
                "example",
                "variables",
                "ta",
                "additional_datasets",
                1,
            ),
            "missing required key 'type'. Needed to find the data of "
            "project 'OBS'.",
        ),
    ]


@pytest.mark.parametrize("project", ["OBS", "OBS6"])
@pytest.mark.parametrize("value", ["sat", ["reanaly", "ground"]])
def test_validate_recipe_known_type(project: str, value: object) -> None:
    recipe = yaml.safe_load(RECIPE)
    recipe["datasets"][1].update(project=project, type=value)
    assert _issues(recipe) == []


def test_validate_recipe_unknown_type() -> None:
    recipe = yaml.safe_load(RECIPE)
    recipe["datasets"][1].update(project="OBS6", type="reanalysis")
    assert _issues(recipe) == [
        RecipeIssue(
            ("datasets", 1, "type"),
            "unknown value 'reanalysis'. Not a type of project 'OBS6', "
            "choose from 'sat', 'reanaly', 'ground', 'clim', or 'campaign'. "
            "Did you mean 'reanaly'?",
            severity="warning",
        ),
    ]


def test_validate_recipe_unknown_type_of_variable_group() -> None:
    recipe = yaml.safe_load(RECIPE)
    variable = recipe["diagnostics"]["example"]["variables"]["ta"]
    variable.update(project="OBS", tier=2, type="atmos")
    variable["additional_datasets"] = [
        {"dataset": "HadISST"},
        {"dataset": "CRU"},
        {"dataset": "GPCP-SG", "type": "sat"},
    ]
    # The type of the variable group is reported once.
    assert _issues(recipe) == [
        RecipeIssue(
            ("diagnostics", "example", "variables", "ta", "type"),
            "unknown value 'atmos'. Not a type of project 'OBS', choose "
            "from 'sat', 'reanaly', 'ground', 'clim', or 'campaign'.",
            severity="warning",
        ),
    ]


def test_validate_recipe_missing_facets_unknown_project() -> None:
    recipe = yaml.safe_load(RECIPE)
    recipe["datasets"][1]["project"] = ["OBS"]
    assert not [i for i in _issues(recipe) if "Needed to find" in i.message]


def test_validate_recipe_missing_facets_esgf_project() -> None:
    recipe = yaml.safe_load(RECIPE)
    del recipe["datasets"][2]["grid"]
    assert _issues(recipe) == [
        RecipeIssue(
            ("datasets", 2),
            "missing required key 'grid'. Needed to find the data of "
            "project 'CMIP6'.",
        ),
    ]


def test_validate_recipe_required_facets_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        CFG["projects"]["CMIP6"],
        "required_facets",
        ["institute"],
    )
    recipe = yaml.safe_load(RECIPE)
    assert _issues(recipe) == [
        RecipeIssue(
            path,
            "missing required key 'institute'. Needed to find the data of "
            "project 'CMIP6'.",
        )
        for path in [
            ("datasets", 2),
            (
                "diagnostics",
                "example",
                "variables",
                "ta",
                "additional_datasets",
                0,
            ),
        ]
    ]


@pytest.mark.parametrize(
    ("dataset", "mip", "missing"),
    [
        ("ERA5", "Amon", []),
        ("ERA5", "Omon", ["type"]),
        ("ERA-Interim", "Amon", ["type"]),
    ],
)
def test_validate_recipe_required_facets_from_extra_facets(
    monkeypatch: pytest.MonkeyPatch,
    dataset: str,
    mip: str,
    missing: list[str],
) -> None:
    monkeypatch.setitem(
        CFG["projects"]["OBS6"],
        "extra_facets",
        {"ERA5": {"A*": {"*": {"type": "reanaly"}}}},
    )
    recipe = yaml.safe_load(RECIPE)
    recipe["datasets"][1] = {
        "dataset": dataset,
        "project": "OBS6",
        "tier": 3,
        "mip": mip,
    }
    issues = [i for i in _issues(recipe) if "Needed to find" in i.message]
    assert [i.path for i in issues] == [("datasets", 1)] * len(missing)


def test_required_facets_of_esgf_projects() -> None:
    """Check that the required facets are used to search ESGF."""
    path = (
        Path(esmvalcore.__file__).parent
        / "config"
        / "configurations"
        / "data-intake-esgf.yml"
    )
    config = yaml.safe_load(path.read_text(encoding="utf-8"))["projects"]
    esgf_facets = {
        project: set(facets)
        for project, facets in esmvalcore.io.esgf.facets.FACETS.items()
    }
    for project, settings in config.items():
        esgf_facets[project] = {
            facet
            for source in settings["data"].values()
            for facet in source["facets"]
        }
    for project, facets in esgf_facets.items():
        required = CFG["projects"][project].get("required_facets", [])
        assert set(required) <= facets, project


@pytest.mark.parametrize(
    "value",
    ["ERA5", "CanESM5", "ERA5|CanESM5", "my-alias", ""],
)
@pytest.mark.parametrize("key", ["reference_dataset", "alternative_dataset"])
def test_validate_recipe_known_reference_dataset(key: str, value: str) -> None:
    recipe = yaml.safe_load(RECIPE)
    recipe["datasets"][0]["alias"] = "my-alias"
    recipe["diagnostics"]["example"]["variables"]["ta"][key] = value
    assert _issues(recipe) == []


@pytest.mark.parametrize("key", ["reference_dataset", "alternative_dataset"])
def test_validate_recipe_unknown_reference_dataset(key: str) -> None:
    recipe = yaml.safe_load(RECIPE)
    recipe["diagnostics"]["example"]["variables"]["ta"][key] = "ERA5|ERA4"
    assert _issues(recipe) == [
        RecipeIssue(
            ("diagnostics", "example", "variables", "ta", key),
            "unknown value 'ERA4'. Not a dataset of this variable group. "
            "Did you mean 'ERA5'?",
            severity="warning",
        ),
    ]


def test_validate_recipe_reference_dataset_of_other_group() -> None:
    recipe = yaml.safe_load(RECIPE)
    variables = recipe["diagnostics"]["example"]["variables"]
    variables["pr"] = {"mip": "Amon", "reference_dataset": "CanESM5"}
    assert [i.path[-2:] for i in _issues(recipe)] == [
        ("pr", "reference_dataset"),
    ]


def test_validate_recipe_reference_dataset_pattern() -> None:
    recipe = yaml.safe_load(RECIPE)
    recipe["datasets"][0]["dataset"] = "*"
    recipe["diagnostics"]["example"]["variables"]["ta"][
        "reference_dataset"
    ] = "X"
    assert _issues(recipe) == []


@pytest.mark.parametrize(
    "ancestors",
    [["ta"], ["example/ta", "example/pr"], ["example/*"], ["*/p?"]],
)
def test_validate_recipe_ancestors(ancestors: list[str]) -> None:
    recipe = yaml.safe_load(RECIPE)
    recipe["diagnostics"]["example"]["scripts"]["plot"]["ancestors"] = (
        ancestors
    )
    assert _issues(recipe) == []


@pytest.mark.parametrize(
    ("ancestor", "suggestion"),
    [
        ("example/tas", " Did you mean 'example/ta'?"),
        ("other/*", ""),
    ],
)
def test_validate_recipe_unknown_ancestor(
    ancestor: str,
    suggestion: str,
) -> None:
    recipe = yaml.safe_load(RECIPE)
    script = recipe["diagnostics"]["example"]["scripts"]["plot"]
    script["ancestors"] = ["ta", ancestor]
    message = (
        f"unknown value {ancestor!r}. Does not match any variable group or "
        f"script.{suggestion}"
    )
    assert _issues(recipe) == [
        RecipeIssue(
            ("diagnostics", "example", "scripts", "plot", "ancestors", 1),
            message,
        ),
    ]


@pytest.mark.parametrize(
    "timerange",
    ["1990/2000", "199001/P5Y", "P2Y/20001231T12", "*", "1990/*", "*/P1Y"],
)
def test_validate_recipe_valid_timerange(timerange: str) -> None:
    dataset = {**SMALL_RECIPE["datasets"][0], "timerange": timerange}
    recipe = {**SMALL_RECIPE, "datasets": [dataset]}
    assert _issues(recipe) == []


@pytest.mark.parametrize(
    ("timerange", "reason"),
    [
        ("2000", "Valid values must be separated by `/`."),
        (
            "P1Y/P2Y",
            "Cannot set both the beginning and the end as duration periods.",
        ),
        ("2000/20x1", "Valid value must follow ISO 8601 standard"),
    ],
)
def test_validate_recipe_invalid_timerange(
    timerange: str,
    reason: str,
) -> None:
    recipe = yaml.safe_load(RECIPE)
    recipe["diagnostics"]["example"]["variables"]["ta"]["timerange"] = (
        timerange
    )
    (issue,) = _issues(recipe)
    assert issue.path == (
        "diagnostics",
        "example",
        "variables",
        "ta",
        "timerange",
    )
    assert issue.message.startswith(f"invalid value {timerange!r}. {reason}")
    assert issue.severity == "error"


OBS6_DATASET = {
    "project": "OBS6",
    "dataset": "X",
    "tier": 2,
    "type": "sat",
    "version": 1,
}


@pytest.mark.parametrize(
    ("variables", "datasets", "message"),
    [
        (
            {"tass": {"mip": "Amon"}},
            None,
            (
                "diagnostics > example > variables > tass: variable 'tass' not "
                "available in table 'Amon' of project 'CMIP6'."
            ),
        ),
        (
            {"tos": {"mip": "Amon"}},
            None,
            (
                "diagnostics > example > variables > tos: variable 'tos' not "
                "available in table 'Amon' of project 'CMIP6'."
            ),
        ),
        (
            {"lwp": {"mip": "Amon"}},
            None,
            (
                "diagnostics > example > variables > lwp: variable 'lwp' not "
                "available in table 'Amon' of project 'CMIP6'."
            ),
        ),
        ({"lwp": {"mip": "Amon", "derive": True}}, None, None),
        ({"x": {"mip": "Amon", "short_name": "tas"}}, None, None),
        ({"tas": {"mip": "A*"}}, None, None),
        # The CMOR checks of OBS6 are not strict, so all tables are used.
        ({"tos": {"mip": "Amon"}}, [OBS6_DATASET], None),
        (
            {"tass": {"mip": "Amon"}},
            [OBS6_DATASET],
            (
                "diagnostics > example > variables > tass: variable 'tass' not "
                "available in table 'Amon' of project 'OBS6'."
            ),
        ),
    ],
)
def test_validate_recipe_unavailable_variable(
    variables: dict[str, Any],
    datasets: list[dict[str, Any]] | None,
    message: str | None,
) -> None:
    diagnostic = {**SMALL_RECIPE["diagnostics"]["example"]}
    diagnostic["variables"] = variables
    recipe = {**SMALL_RECIPE, "diagnostics": {"example": diagnostic}}
    if datasets is not None:
        recipe["datasets"] = datasets
    expected = [] if message is None else [message]
    assert [str(i) for i in _issues(recipe)] == expected


@pytest.mark.parametrize(
    ("path", "value", "message", "severity"),
    [
        (
            ("datasets", 0, "mip"),
            "atmos",
            "unknown value 'atmos'. Not a CMOR table of project 'CMIP5'.",
            "error",
        ),
        (
            ("datasets", 2, "mip"),
            "Amonn",
            (
                "unknown value 'Amonn'. Not a CMOR table of project 'CMIP6'. "
                "Did you mean 'Amon'?"
            ),
            "error",
        ),
        (
            ("datasets", 1, "mip"),
            "Lyr",
            (
                "unknown value 'Lyr'. Not a CMOR table of project 'native6'. "
                "The variable will be looked up in all tables, because the "
                "CMOR checks of this project are not strict. Did you mean "
                "'Oyr'?"
            ),
            "warning",
        ),
    ],
)
def test_validate_recipe_mip_in_dataset(
    path: tuple[str | int, ...],
    value: str,
    message: str,
    severity: Literal["error", "warning"],
) -> None:
    recipe = yaml.safe_load(RECIPE)
    recipe["datasets"][path[1]]["mip"] = value
    issues = [i for i in _issues(recipe) if i.path[-1] == "mip"]
    assert issues == [RecipeIssue(path, message, severity=severity)]


def test_validate_recipe_mip_in_variable_group() -> None:
    # The project is defined in another section than the MIP.
    recipe = yaml.safe_load(RECIPE)
    recipe["diagnostics"]["example"]["variables"]["ta"]["mip"] = "Amonn"
    path = ("diagnostics", "example", "variables", "ta", "mip")
    issues = [i for i in _issues(recipe) if i.path == path]
    start = "unknown value 'Amonn'. Not a CMOR table of project "
    assert [(i.message[: len(start) + 6], i.severity) for i in issues] == [
        (f"{start}'CMIP5", "error"),
        (f"{start}'nativ", "warning"),
        (f"{start}'CMIP6", "error"),
    ]
    assert "Did you mean 'Amon'?" in issues[2].message


def test_validate_recipe_mip_reported_once() -> None:
    recipe = copy.deepcopy(SMALL_RECIPE)
    recipe["diagnostics"]["example"]["variables"] = {
        "tas": {"mip": "Amonn"},
        "pr": {"mip": "Amonn"},
    }
    recipe["datasets"].append({**recipe["datasets"][0], "dataset": "B"})
    paths = [i.path for i in _issues(recipe)]
    assert paths == [
        ("diagnostics", "example", "variables", "tas", "mip"),
        ("diagnostics", "example", "variables", "pr", "mip"),
    ]


def test_validate_recipe_mip_not_strict_project() -> None:
    recipe = copy.deepcopy(SMALL_RECIPE)
    recipe["datasets"] = [OBS6_DATASET]
    recipe["diagnostics"]["example"]["variables"] = {"tas": {"mip": "Amonn"}}
    (issue,) = _issues(recipe)
    assert issue.severity == "warning"
    assert issue.path == ("diagnostics", "example", "variables", "tas", "mip")
    assert "Not a CMOR table of project 'OBS6'. The variable will be" in (
        issue.message
    )


def test_validate_recipe_mip_of_overridden_project() -> None:
    # The dataset changes the project of the variable group, but not the MIP.
    recipe = copy.deepcopy(SMALL_RECIPE)
    recipe["diagnostics"]["example"]["variables"] = {
        "tas": {"mip": "Amon", "project": "CMIP6"},
    }
    recipe["diagnostics"]["example"]["additional_datasets"] = [
        {"dataset": "B", "project": "CMIP7"},
    ]
    issues = [i for i in _issues(recipe) if "CMOR table" in i.message]
    assert [(i.path, i.severity) for i in issues] == [
        (("diagnostics", "example", "additional_datasets", 0), "error"),
    ]
    assert issues[0].message.startswith(
        "MIP 'Amon' of the variable group is not a CMOR table of project "
        "'CMIP7'.",
    )


@pytest.mark.parametrize("mip", ["*", "A*", "Amo?", "[AO]mon", 1])
def test_validate_recipe_mip_skipped(mip: str | int) -> None:
    recipe = copy.deepcopy(SMALL_RECIPE)
    recipe["diagnostics"]["example"]["variables"] = {"tas": {"mip": mip}}
    messages = [i.message for i in _issues(recipe)]
    assert not any("CMOR table" in m for m in messages)


def test_validate_recipe_mip_project_without_tables(session: Session) -> None:
    session["projects"]["CMIP6"]["cmor_table"] = {
        "type": "esmvalcore.cmor.table.NoInfo",
    }
    recipe = copy.deepcopy(SMALL_RECIPE)
    recipe["diagnostics"]["example"]["variables"] = {"tas": {"mip": "Amonn"}}
    assert _issues(recipe, session) == []


def _align_metadata_messages(**settings: Any) -> list[str]:
    recipe = copy.deepcopy(SMALL_RECIPE)
    recipe["preprocessors"] = {"pp": {"align_metadata": settings}}
    recipe["diagnostics"]["example"]["variables"] = {
        "tas": {"mip": "Amon", "preprocessor": "pp"},
    }
    return [str(issue) for issue in _issues(recipe)]


@pytest.mark.parametrize(
    "settings",
    [
        {"target_project": "CMIP5"},
        {"target_project": "CMIP6", "target_short_name": "pr"},
        {"target_project": "CMIP6", "target_mip": "day"},
        {
            "target_project": "CMIP7",
            "target_mip": "atmos",
            "target_branding_suffix": "tavg-h2m-hxy-u",
        },
        {
            "target_project": "CMIP6",
            "target_short_name": "zzz",
            "strict": False,
        },
        {"target_project": "CMIP6", "target_short_name": "z*"},
        # The same rule as for datasets: CORDEX tables are named 3h, not 3hr.
        {
            "target_project": "CORDEX",
            "target_mip": "3hr",
            "target_short_name": "tas",
        },
    ],
)
def test_validate_recipe_align_metadata_valid(
    settings: dict[str, Any],
) -> None:
    assert _align_metadata_messages(**settings) == []


@pytest.mark.parametrize("strict", [True, False])
def test_validate_recipe_align_metadata_unknown_project(strict: bool) -> None:
    messages = _align_metadata_messages(target_project="CMIP66", strict=strict)
    message = (
        "preprocessors > pp > align_metadata > target_project: unknown value "
        "'CMIP66'. Not a configured project. Did you mean 'CMIP6'?"
    )
    assert messages == [message]


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        (
            {"target_short_name": "zzz", "target_mip": "Amon"},
            (
                "target variable 'zzz' not available in table 'Amon' of "
                "project 'CMIP6'."
            ),
        ),
        (
            {"target_short_name": "zzz"},
            (
                "target variable 'zzz' not available in table 'Amon' of "
                "project 'CMIP6'. The target MIP is taken from diagnostics > "
                "example > variables > tas."
            ),
        ),
        (
            {"target_mip": "Omon"},
            (
                "target variable 'tas' not available in table 'Omon' of "
                "project 'CMIP6'. The target short name is taken from "
                "diagnostics > example > variables > tas."
            ),
        ),
    ],
)
def test_validate_recipe_align_metadata_unavailable(
    settings: dict[str, Any],
    message: str,
) -> None:
    messages = _align_metadata_messages(target_project="CMIP6", **settings)
    assert messages == [
        (
            f"preprocessors > pp > align_metadata: {message} Set `strict: "
            "false` to ignore this."
        ),
    ]


def _cmip7_recipe(**facets: Any) -> dict[str, Any]:
    recipe = copy.deepcopy(SMALL_RECIPE)
    recipe["datasets"] = [
        {
            "project": "CMIP7",
            "dataset": "X",
            "exp": "historical",
            "ensemble": "r1i1p1f1",
            "grid": "g",
            "region": "glb",
            "frequency": "mon",
            "activity": "CMIP",
            "institute": "Y",
            "version": "v1",
        },
    ]
    variable = {"mip": "atmos", "branding_suffix": "tavg-h2m-hxy-u", **facets}
    recipe["diagnostics"]["example"]["variables"] = {"tas": variable}
    return recipe


def test_validate_recipe_branding_suffix() -> None:
    assert _issues(_cmip7_recipe()) == []


def test_validate_recipe_branding_suffix_invalid() -> None:
    recipe = _cmip7_recipe(branding_suffix="tavg-h2m-hxy-uu")
    (issue,) = _issues(recipe)
    assert issue.path == ("diagnostics", "example", "variables", "tas")
    assert issue.message.startswith(
        "variable 'tas' with branding suffix 'tavg-h2m-hxy-uu' not available "
        "in table 'atmos' of project 'CMIP7'. Available branding suffixes "
        "are: ",
    )
    assert issue.message.endswith(" Did you mean 'tavg-h2m-hxy-u'?")


@pytest.mark.parametrize("branding_suffix", ["tavg-*", "tavg-h2m-hxy-[ul]"])
def test_validate_recipe_branding_suffix_pattern(branding_suffix: str) -> None:
    assert _issues(_cmip7_recipe(branding_suffix=branding_suffix)) == []


def _recipe_with_script(script: object) -> dict[str, Any]:
    recipe = yaml.safe_load(RECIPE)
    recipe["documentation"]["authors"] = AUTHORS[:1]
    recipe["diagnostics"]["example"]["scripts"]["plot"]["script"] = script
    return recipe


@pytest.mark.usefixtures("esmvaltool")
def test_validate_recipe_script_in_working_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "my_diagnostic.py").write_text("")
    monkeypatch.chdir(tmp_path)
    assert _issues(_recipe_with_script("my_diagnostic.py")) == []


def test_validate_recipe_missing_script(esmvaltool: Diagnostics) -> None:
    recipe = _recipe_with_script("examples/diagnostc.py")
    assert _issues(recipe) == [
        RecipeIssue(
            ("diagnostics", "example", "scripts", "plot", "script"),
            "file not found. 'examples/diagnostc.py' is not a file in the "
            "working directory or in the diagnostic scripts of ESMValTool "
            f"in {esmvaltool.scripts}.",
        ),
    ]


@pytest.mark.usefixtures("esmvaltool")
def test_validate_recipe_missing_script_without_diagnostics(
    session: Session,
) -> None:
    session["run_diagnostic"] = False
    recipe = _recipe_with_script("examples/diagnostc.py")
    (issue,) = _issues(recipe, session=session)
    assert issue.message.startswith("file not found.")
    assert issue.severity == "warning"


def test_validate_recipe_script_without_esmvaltool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(validate, "DIAGNOSTICS", Diagnostics(tmp_path))
    recipe = _recipe_with_script("examples/diagnostc.py")
    assert _issues(recipe) == []


@pytest.mark.usefixtures("esmvaltool")
def test_validate_recipe_script_not_a_string() -> None:
    recipe = _recipe_with_script(1)
    assert [i.message for i in _issues(recipe)] == [
        "expected a string, got 1.",
    ]


def test_ensure_valid_recipe(tmp_path: Path) -> None:
    path = tmp_path / "recipe_test.yml"
    msg = (
        f"Recipe {path} is invalid:\n"
        "- line 5: datasets > 1: missing required key 'dataset'.\n"
        "- line 5: datasets > 1 > tier: expected at most 3, got 7.\n"
    )
    with pytest.raises(RecipeError, match=re.escape(msg)):
        ensure_valid_recipe(INVALID_RECIPE, path)


def test_ensure_valid_recipe_invalid_yaml(tmp_path: Path) -> None:
    path = tmp_path / "recipe_test.yml"
    msg = f"Recipe {path} is invalid:\n- line 3: recipe: invalid YAML: "
    with pytest.raises(RecipeError, match=re.escape(msg)):
        ensure_valid_recipe("documentation:\n  title: [\n", path)


@pytest.mark.usefixtures("esmvaltool")
def test_ensure_valid_recipe_warnings(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING)
    recipe = yaml.safe_load(RECIPE)
    recipe["documentation"]["authors"] = ["author_100"]
    path = tmp_path / "recipe_test.yml"
    assert ensure_valid_recipe(yaml.safe_dump(recipe), path) == recipe
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    message = caplog.records[0].getMessage()
    assert message.startswith(f"Recipe {path}: line ")
    assert (
        "documentation > authors > 0: unknown value 'author_100'." in message
    )


def test_ensure_valid_recipe_unknown_key(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING)
    recipe = yaml.safe_load(RECIPE)
    recipe["documentation"]["domains"] = ["global"]
    path = tmp_path / "recipe_test.yml"
    ensure_valid_recipe(yaml.safe_dump(recipe), path)
    (message,) = [r.getMessage() for r in caplog.records]
    assert message.startswith(f"Recipe {path}: line ")
    assert message.endswith(
        ": documentation > domains: unknown key. Unknown keys are ignored, "
        "this is deprecated and will not work in v2.18.0.",
    )


def test_ensure_valid_recipe_duplicate_keys(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING)
    path = tmp_path / "recipe_test.yml"
    ensure_valid_recipe(DUPLICATE_RECIPE, path)
    assert [r.levelname for r in caplog.records] == ["WARNING", "WARNING"]
    assert caplog.records[1].getMessage() == (
        f"Recipe {path}: line 16: diagnostics > example > variables > tas: "
        "duplicate key, this definition is ignored because 'tas' is defined "
        "again on line 18."
    )


def test_ensure_valid_datasets_warnings(
    session: Session,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING)
    recipe = yaml.safe_load(RECIPE)
    recipe["datasets"][1]["mip"] = "Lyr"
    validate.ensure_valid_datasets(recipe, session)
    (message,) = [r.getMessage() for r in caplog.records]
    assert message.startswith(
        "Recipe: datasets > 1 > mip: unknown value 'Lyr'.",
    )
