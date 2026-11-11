"""Tests for :mod:`esmvalcore._recipe.json_schema`."""

from __future__ import annotations

import inspect
import logging
import re
import typing
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol

import jsonschema
import pytest
import yaml
from iris.cube import Cube

import esmvalcore.io.esgf.facets
import esmvalcore.preprocessor
from esmvalcore._recipe import check, json_schema
from esmvalcore._recipe.json_schema import (
    RecipeIssue,
    _annotation_to_schema,
    _clean_rst,
    _parse_docstring,
    _resolve_annotation,
    _rst_to_text,
    generate_recipe_schema,
    validate_recipe,
)
from esmvalcore.config import CFG
from esmvalcore.config._diagnostics import Diagnostics, TagsManager
from esmvalcore.exceptions import RecipeError

type Scheme = Literal["linear", "nearest"]


class SupportsExample(Protocol):
    def example(self) -> None: ...


@pytest.mark.parametrize(
    ("annotation", "expected"),
    [
        (str, {"type": "string"}),
        (int, {"type": "integer"}),
        (float, {"type": "number"}),
        (bool, {"type": "boolean"}),
        (None, {"type": "null"}),
        (Path, {"type": "string"}),
        (Literal["a", "b"], {"enum": ["a", "b"]}),
        (Scheme, {"enum": ["linear", "nearest"]}),
        (
            str | None,
            {"anyOf": [{"type": "string"}, {"type": "null"}]},
        ),
        (list[str], {"type": "array", "items": {"type": "string"}}),
        (Sequence[float], {"type": "array", "items": {"type": "number"}}),
        (Iterable[str], {"type": "array", "items": {"type": "string"}}),
        (tuple[int, ...], {"type": "array", "items": {"type": "integer"}}),
        (
            tuple[int, str],
            {
                "type": "array",
                "items": [{"type": "integer"}, {"type": "string"}],
                "minItems": 2,
                "maxItems": 2,
            },
        ),
        (tuple, {"type": "array"}),
        (list[Any], {"type": "array"}),
        (dict[str, Any], {"type": "object"}),
        (Any, None),
        (json_schema._Unknown, None),
        (str | json_schema._Unknown, None),
        (Cube, None),
        (str | Cube, {"type": "string"}),
        (Cube | None, {"type": "null"}),
        (
            Cube | Path | str | dict,
            {"anyOf": [{"type": "string"}, {"type": "object"}]},
        ),
        (str | SupportsExample, None),
    ],
)
def test_annotation_to_schema(
    annotation: object,
    expected: dict[str, Any] | None,
) -> None:
    assert _annotation_to_schema(annotation) == expected


def test_resolve_annotation() -> None:
    # The preprocessor functions only import `Cube` for type checking.
    function = esmvalcore.preprocessor.extract_region
    unknown = json_schema._Unknown
    assert _resolve_annotation("Sequence[float]", function) == Sequence[float]
    assert _resolve_annotation("Cube", function) is Cube
    assert typing.get_args(_resolve_annotation("Cube | None", function)) == (
        Cube,
        type(None),
    )
    assert _resolve_annotation("Undefined", function) is unknown
    assert _resolve_annotation("Undefined.attribute", function) is unknown
    assert _resolve_annotation("str(", function) is unknown
    assert _resolve_annotation(int, function) is int


def test_parse_docstring() -> None:
    docstring = """Do something useful.

    More details that are not part of the summary.

    Parameters
    ----------
    cube:
        Input cube.
    target : str
        Use :func:`esmvalcore.preprocessor.regrid` for
        ``details``.

        Second paragraph.
    **kwargs:
        Other arguments.

    Returns
    -------
    iris.cube.Cube
        Output cube.
    """
    summary, descriptions = _parse_docstring(
        "\n".join(line.removeprefix("    ") for line in docstring.split("\n")),
    )
    assert summary == "Do something useful."
    assert descriptions == {
        "cube": "Input cube.",
        "target": (
            "Use :func:`esmvalcore.preprocessor.regrid` for\n``details``.\n\n"
            "Second paragraph."
        ),
        "kwargs": "Other arguments.",
    }
    assert _rst_to_text(descriptions["target"]) == (
        "Use esmvalcore.preprocessor.regrid for `details`.\n\n"
        "Second paragraph."
    )


def test_rst_to_text_code_block() -> None:
    text = (
        "For example:\n\n"
        ".. code-block:: python\n\n"
        "    scheme = {\n"
        '        "reference": "iris.analysis:Linear",\n'
        "    }\n\n"
        "Last\nparagraph."
    )
    code = '{\n    "reference": "iris.analysis:Linear",\n}'
    assert _rst_to_text(text) == (
        "For example:\n\n"
        "    scheme = {\n"
        '        "reference": "iris.analysis:Linear",\n'
        "    }\n\n"
        "Last paragraph."
    )
    assert _rst_to_text(text, markdown=True) == (
        f"For example:\n\n```python\nscheme = {code}\n```\n\nLast paragraph."
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            ":func:`~esmvalcore.preprocessor.regrid`",
            "esmvalcore.preprocessor.regrid",
        ),
        (":ref:`this table <supported_stat_operator>`", "this table"),
        (":math:`< 10^{-6}` kg m :math:`^{-2}`", "< 10^{-6} kg m ^{-2}"),
        ("``value``", "`value`"),
    ],
)
def test_clean_rst(text: str, expected: str) -> None:
    assert _clean_rst(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            ":ref:`default regridding schemes`",
            (
                "[Default regridding schemes](https://docs.esmvaltool.org/"
                "projects/ESMValCore/en/latest/recipe/preprocessor.html"
                "#default-regridding-schemes)"
            ),
        ),
        (
            ":ref:`regrid preprocessor <Horizontal regridding>`",
            (
                "[regrid preprocessor](https://docs.esmvaltool.org/projects/"
                "ESMValCore/en/latest/recipe/preprocessor.html"
                "#horizontal-regridding)"
            ),
        ),
        (":ref:`unknown_label`", "unknown_label"),
        (":ref:`text <unknown_label>`", "text"),
    ],
)
def test_clean_rst_markdown(text: str, expected: str) -> None:
    assert _clean_rst(text, markdown=True) == expected


def test_facet_markdown_description(schema: dict[str, Any]) -> None:
    mip = schema["definitions"]["facets"]["properties"]["mip"]
    assert mip["markdownDescription"] == (
        "MIP table, for example `Amon` for CMIP6 or `atmos` for CMIP7."
    )
    assert "`" not in mip["description"]


def test_step_schema_markdown_description(schema: dict[str, Any]) -> None:
    steps = schema["definitions"]["steps"]
    scheme = steps["regrid"]["anyOf"][0]["properties"]["scheme"]
    assert (
        "[Generic regridding schemes](https://"
        in (scheme["markdownDescription"])
    )
    assert "```python\n" in scheme["markdownDescription"]
    assert "(described in Generic regridding schemes)" in scheme["description"]
    target = steps["extract_levels"]["anyOf"][0]["properties"]["levels"]
    assert "markdownDescription" not in target


@pytest.fixture(scope="module")
def schema() -> dict[str, Any]:
    return generate_recipe_schema()


def test_schema_is_valid(schema: dict[str, Any]) -> None:
    jsonschema.Draft7Validator.check_schema(schema)


def test_schema_has_all_preprocessor_steps(schema: dict[str, Any]) -> None:
    steps = schema["definitions"]["steps"]
    assert list(steps) == list(json_schema.RECIPE_STEPS)
    for step, step_schema in steps.items():
        assert step_schema["description"], step


def test_step_schema(schema: dict[str, Any]) -> None:
    settings = schema["definitions"]["steps"]["extract_levels"]["anyOf"][0]
    assert settings["required"] == ["levels", "scheme"]
    assert settings["additionalProperties"] is False
    assert settings["properties"]["scheme"]["enum"] == [
        "linear",
        "nearest",
        "linear_extrapolate",
        "nearest_extrapolate",
    ]
    assert settings["properties"]["rtol"]["default"] == 1e-7
    assert settings["properties"]["scheme"]["description"]


def test_step_schema_recipe_arguments(schema: dict[str, Any]) -> None:
    steps = schema["definitions"]["steps"]
    target_grid = steps["regrid"]["anyOf"][0]["properties"]["target_grid"]
    assert target_grid == json_schema.RECIPE_ARGUMENTS["regrid"]["target_grid"]
    assert "Dataset" not in target_grid["description"]


def test_step_schema_statistics_operators(schema: dict[str, Any]) -> None:
    steps = schema["definitions"]["steps"]
    operator = steps["climate_statistics"]["anyOf"][0]["properties"][
        "operator"
    ]
    assert operator["enum"] == list(json_schema.STATISTICS_OPERATORS)
    assert operator["default"] == "mean"
    assert "this table" not in operator["description"]
    assert "Allowed options are `gmean`, `hmean`" in operator["description"]


@pytest.mark.parametrize(
    ("statistics", "path", "message"),
    [
        (["mean", "MEAN"], (1,), "unknown value 'MEAN'."),
        (
            [{"operator": "percentile", "percent": 5}, {"operator": "meen"}],
            (1, "operator"),
            "unknown value 'meen'.",
        ),
        ([{"percent": 5}], (0,), "missing required key 'operator'."),
        (
            [{"operator": "percentile"}],
            (0,),
            "missing required key 'percent'.",
        ),
        (
            [{"operator": "percentile", "percent": 105}],
            (0, "percent"),
            "expected at most 100, got 105.",
        ),
    ],
)
def test_validate_recipe_statistics_operators(
    statistics: list[str | dict[str, Any]],
    path: tuple[str | int, ...],
    message: str,
) -> None:
    recipe = yaml.safe_load(RECIPE)
    recipe["preprocessors"]["pp"]["multi_model_statistics"] = {
        "span": "overlap",
        "statistics": statistics,
    }
    issues = validate_recipe(recipe)
    assert len(issues) == 1
    issue = issues[0]
    prefix = ("preprocessors", "pp", "multi_model_statistics", "statistics")
    assert issue.path == (*prefix, *path)
    assert issue.message.startswith(message)
    assert issue.severity == "error"


@pytest.mark.parametrize(
    ("settings", "issues"),
    [
        ({"operator": "mean"}, []),
        ({"operator": "percentile", "percent": 95}, []),
        ({"operator": "wpercentile", "percent": [5, 95]}, []),
        ({"operator": "percentile"}, ["missing required argument 'percent'."]),
    ],
)
def test_validate_recipe_percent(
    settings: dict[str, Any],
    issues: list[str],
) -> None:
    recipe = yaml.safe_load(RECIPE)
    recipe["preprocessors"]["pp"]["annual_statistics"] = settings
    result = validate_recipe(recipe)
    assert [issue.message for issue in result] == issues
    path = ("preprocessors", "pp", "annual_statistics")
    assert all(issue.path == path for issue in result)


@pytest.mark.parametrize(
    ("step", "settings"),
    [
        ("regrid", {"target_grid": "ERA5", "scheme": "linear"}),
        ("regrid", {"target_grid": "reference_dataset", "scheme": "linear"}),
        ("regrid", {"target_grid": "EUR-11", "scheme": "linear"}),
        ("regrid", {"target_grid": "2.5x2.5", "scheme": "linear"}),
        (
            "regrid",
            {
                "target_grid": {
                    "start_longitude": 0,
                    "end_longitude": 10,
                    "step_longitude": 1,
                    "start_latitude": 40,
                    "end_latitude": 50,
                    "step_latitude": 1,
                },
                "scheme": "linear",
            },
        ),
        ("extract_levels", {"levels": 85000, "scheme": "linear"}),
        ("extract_levels", {"levels": [85000, 50000], "scheme": "linear"}),
        ("extract_levels", {"levels": "ERA5", "scheme": "linear"}),
        (
            "extract_levels",
            {"levels": {"dataset": "ERA5"}, "scheme": "linear"},
        ),
        (
            "extract_levels",
            {
                "levels": {"cmor_table": "CMIP6", "coordinate": "plev19"},
                "scheme": "linear",
            },
        ),
        ("extract_shape", {"shapefile": "ar6", "ids": {"Acronym": ["WNA"]}}),
    ],
)
def test_validate_recipe_arguments(step: str, settings: dict) -> None:
    recipe = yaml.safe_load(RECIPE)
    recipe["preprocessors"]["test"] = {step: settings}
    assert validate_recipe(recipe) == []


@pytest.mark.parametrize(
    ("step", "settings", "message"),
    [
        (
            "regrid",
            {"target_grid": {"start_longitude": 0}, "scheme": "linear"},
            (
                "preprocessors > test > regrid > target_grid: missing "
                "required keys 'end_longitude', 'step_longitude', "
                "'start_latitude', 'end_latitude', 'step_latitude'."
            ),
        ),
        (
            "extract_levels",
            {"levels": {"cmor_table": "CMIP6"}, "scheme": "linear"},
            (
                "preprocessors > test > extract_levels > levels: missing required "
                "key 'coordinate'."
            ),
        ),
        (
            "extract_levels",
            {"levels": True, "scheme": "linear"},
            (
                "preprocessors > test > extract_levels > levels: expected a "
                "number, a list, a string, or a mapping, got True."
            ),
        ),
    ],
)
def test_validate_recipe_arguments_invalid(
    step: str,
    settings: dict,
    message: str,
) -> None:
    recipe = yaml.safe_load(RECIPE)
    recipe["preprocessors"]["test"] = {step: settings}
    assert message in [str(issue) for issue in validate_recipe(recipe)]


@pytest.mark.parametrize(
    ("definition", "key", "target"),
    [
        ("documentation", None, "overview.html#recipe-documentation"),
        (None, "datasets", "overview.html#datasets"),
        (None, "preprocessors", "overview.html#preprocessors"),
        (None, "diagnostics", "overview.html#diagnostics"),
        ("documentation", "authors", "overview.html#recipe-documentation"),
        ("documentation", "maintainer", "overview.html#recipe-documentation"),
        ("documentation", "projects", "overview.html#recipe-documentation"),
        ("documentation", "references", "overview.html#recipe-documentation"),
        ("diagnostic", "themes", "overview.html#diagnostics"),
        ("diagnostic", "realms", "overview.html#diagnostics"),
        (
            "diagnostic",
            "variables",
            "overview.html#variable-and-dataset-definitions",
        ),
        ("diagnostic", "scripts", "overview.html#running-a-simple-diagnostic"),
        (
            "diagnostic",
            "additional_datasets",
            "overview.html#diagnostic-and-variable-specific-datasets",
        ),
        ("script", "script", "overview.html#running-a-simple-diagnostic"),
        ("script", "ancestors", "overview.html#ancestor-tasks"),
        ("variable", "preprocessor", "overview.html#preprocessors"),
        (
            "variable",
            "additional_datasets",
            "overview.html#diagnostic-and-variable-specific-datasets",
        ),
        ("preprocessor", "custom_order", "overview.html#preprocessors"),
        ("facets", "alias", "overview.html#dataset-alias"),
        (
            "facets",
            "ensemble",
            (
                "overview.html"
                "#short-notation-of-ensemble-members-and-sub-experiments"
            ),
        ),
        ("facets", "domain", "overview.html#cordex-datasets"),
        ("facets", "driver", "overview.html#cordex-datasets"),
        ("facets", "rcm_version", "overview.html#cordex-datasets"),
        ("facets", "timerange", "overview.html#timerange-examples"),
        ("facets", "derive", "preprocessor.html#derive"),
        ("facets", "force_derivation", "preprocessor.html#derive"),
        (
            "facets",
            "supplementary_variables",
            "overview.html#supplementary-variables",
        ),
    ],
)
def test_recipe_item_documentation_link(
    schema: dict[str, Any],
    definition: str | None,
    key: str | None,
    target: str,
) -> None:
    item = schema if definition is None else schema["definitions"][definition]
    if key is not None:
        item = item["properties"][key]
    assert item["description"]
    assert item["markdownDescription"].endswith(
        "(https://docs.esmvaltool.org/projects/ESMValCore/en/latest/recipe/"
        f"{target}).",
    )
    assert "\n\nRefer to [" in item["markdownDescription"]


def test_documentation_link_keeps_markdown(schema: dict[str, Any]) -> None:
    timerange = schema["definitions"]["facets"]["properties"]["timerange"]
    assert "`" not in timerange["description"]
    assert "`1990/2000`" in timerange["markdownDescription"]


def test_step_schema_documentation_link(schema: dict[str, Any]) -> None:
    steps = schema["definitions"]["steps"]
    assert steps["regrid"]["markdownDescription"].endswith(
        "Refer to [how to use `regrid` in a recipe](https://docs.esmvaltool."
        "org/projects/ESMValCore/en/latest/recipe/preprocessor.html#regrid).",
    )
    assert steps["save"]["markdownDescription"].endswith(
        "preprocessor.html#save).",
    )


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("2.12.0", "v2.12.0"),
        ("2.15.0rc5", "latest"),
        ("2.15.0.dev12+g1234567", "latest"),
    ],
)
def test_recipe_documentation_url(
    monkeypatch: pytest.MonkeyPatch,
    version: str,
    expected: str,
) -> None:
    monkeypatch.setattr(json_schema, "__version__", version)
    assert json_schema._recipe_documentation_url("extract_levels") == (
        f"https://docs.esmvaltool.org/projects/ESMValCore/en/{expected}/"
        "recipe/preprocessor.html#extract-levels"
    )


PREPROCESSOR_DOCS = (
    Path(__file__).parents[3] / "doc" / "recipe" / "preprocessor.rst"
)


@pytest.mark.skipif(
    not PREPROCESSOR_DOCS.exists(),
    reason="documentation not available",
)
def test_recipe_documentation_anchors() -> None:
    """Check that the links to the recipe documentation of the steps work.

    Sphinx creates the anchor ``extract-levels`` from the section title
    ````extract_levels```` and from the label ``.. _extract_levels:``.
    """
    text = PREPROCESSOR_DOCS.read_text(encoding="utf-8")
    titles = set(re.findall(r"^``(\w+)``\n[=\-~^*]{4,}$", text, re.MULTILINE))
    labels = set(re.findall(r"^\.\. _(\w+):$", text, re.MULTILINE))
    assert set(json_schema.RECIPE_STEPS) - titles - labels == set()


DOC_DIR = Path(__file__).parents[3] / "doc"


@pytest.mark.skipif(not DOC_DIR.exists(), reason="documentation not available")
def test_documentation_labels() -> None:
    """Check that the labels referred to in the docstrings are known."""
    docstrings = "\n".join(
        inspect.getdoc(getattr(esmvalcore.preprocessor, step)) or ""
        for step in json_schema.RECIPE_STEPS
    )
    references = re.findall(
        r":ref:`(?:[^`<]+?\s+<([^`>]+)>|([^`]+))`",
        docstrings,
    )
    labels = {(a or b).lower() for a, b in references}
    assert labels - set(json_schema.DOCUMENTATION_LABELS) == set()

    for label, (document, title) in json_schema.DOCUMENTATION_LABELS.items():
        text = (DOC_DIR / f"{document}.rst").read_text(encoding="utf-8")
        match = re.search(
            rf"^\.\. _{re.escape(label)}:\n\n(.*)\n(.*)$",
            text,
            re.MULTILINE | re.IGNORECASE,
        )
        assert match, label
        if title is not None:
            assert match[1] == title, label
            assert re.fullmatch(r"([=\-~^*])\1+", match[2]), label


def test_step_schema_literal_arguments(schema: dict[str, Any]) -> None:
    steps = schema["definitions"]["steps"]
    extract_shape = steps["extract_shape"]["anyOf"][0]["properties"]
    assert extract_shape["method"]["enum"] == ["contains", "representative"]
    for step in ("ensemble_statistics", "multi_model_statistics"):
        span = steps[step]["anyOf"][0]["properties"]["span"]
        assert span["enum"] == ["overlap", "full"]


def test_step_schema_descriptions(schema: dict[str, Any]) -> None:
    axis_statistics = schema["definitions"]["steps"]["axis_statistics"]
    properties = axis_statistics["anyOf"][0]["properties"]
    assert all(p.get("description") for p in properties.values())


# Arguments of preprocessor functions that accept any value in a recipe.
UNTYPED_ARGUMENTS = {
    # The type `ArrayLike` accepts numbers and lists of numbers.
    ("extract_point", "latitude"),
    ("extract_point", "longitude"),
}


@pytest.mark.parametrize("step", json_schema.RECIPE_STEPS)
def test_step_schema_arguments_documented(
    schema: dict[str, Any],
    step: str,
) -> None:
    """Check that the arguments of each step have a type and description.

    An argument without a type hint that can be converted to a JSON Schema
    accepts any value in a recipe.
    """
    settings = schema["definitions"]["steps"][step]["anyOf"][0]
    for name, argument in settings["properties"].items():
        assert argument.get("description"), f"{step}: {name}"
        constraints = set(argument) - {
            "default",
            "description",
            "markdownDescription",
        }
        if (step, name) not in UNTYPED_ARGUMENTS:
            assert constraints, f"{step}: {name}"


def test_untyped_arguments_exist(schema: dict[str, Any]) -> None:
    steps = schema["definitions"]["steps"]
    for step, name in UNTYPED_ARGUMENTS:
        assert name in steps[step]["anyOf"][0]["properties"], (step, name)


@pytest.mark.parametrize(
    "constant",
    ["AUTOMATIC_ARGUMENTS", "INTERNAL_ARGUMENTS", "RECIPE_ARGUMENTS"],
)
def test_argument_constants_match_functions(constant: str) -> None:
    """Check that the constants refer to existing arguments.

    This catches renamed or removed arguments, but not new arguments that
    ESMValCore fills in, which must be added to the constants by hand. The
    argument ``session`` is listed for all final steps, whether they have it
    or not.
    """
    for step, names in getattr(json_schema, constant).items():
        function = getattr(esmvalcore.preprocessor, step)
        parameters = inspect.signature(function).parameters
        assert set(names) - {"session"} <= set(parameters), step


def test_exclude_steps_match_functions() -> None:
    assert set(json_schema.RECIPE_STEPS) >= json_schema.EXCLUDE_STEPS


@pytest.mark.skipif(
    not PREPROCESSOR_DOCS.exists(),
    reason="documentation not available",
)
def test_statistics_operators_documented() -> None:
    """Check that the operators match the table in the documentation."""
    text = PREPROCESSOR_DOCS.read_text(encoding="utf-8")
    table = text.split(".. _supported_stat_operator:", 1)[1].split("\n\n")[1]
    operators = re.findall(r"^``(\w+)``", table, re.MULTILINE)
    assert tuple(operators) == json_schema.STATISTICS_OPERATORS


def test_step_schema_internal_arguments(schema: dict[str, Any]) -> None:
    steps = schema["definitions"]["steps"]
    mm_stats = steps["multi_model_statistics"]["anyOf"][0]
    assert "output_products" not in mm_stats["properties"]
    assert "exclude" in mm_stats["properties"]
    assert steps["derive"]["anyOf"][0]["properties"] == {}
    assert "reference" not in steps["bias"]["anyOf"][0]["properties"]
    save = steps["save"]["anyOf"][0]
    assert set(save["properties"]) == {"optimize_access", "compress", "alias"}
    assert "required" not in save


def test_step_schema_automatic_arguments(schema: dict[str, Any]) -> None:
    steps = schema["definitions"]["steps"]
    regrid_time = steps["regrid_time"]["anyOf"][0]
    assert "frequency" in regrid_time["properties"]
    assert "frequency" not in regrid_time.get("required", [])


RECIPE = """
documentation:
  title: Example recipe
  description: Example recipe.
  authors: [andela_bouwe]

datasets:
  - {dataset: CanESM2, project: CMIP5, exp: historical, ensemble: r1i1p1}
  - {dataset: ERA5, project: native6, tier: 3, type: reanaly, version: 1}
  - dataset: ACCESS-CM2
    project: CMIP6
    exp: historical
    ensemble: r1i1p1f1
    grid: gn
    activity: [CMIP, DCPP]

preprocessors:
  default: {}
  pp:
    custom_order: true
    derive:
    extract_levels:
      levels: 85000
      scheme: nearest
    regrid:
      target_grid: 1x1
      scheme: linear
    area_statistics:
      operator: mean
    multi_model_statistics:
      span: overlap
      statistics: [mean]
      exclude: [reference_dataset]
    convert_units:
      units: degC
    mask_landsea: false
    remove_supplementary_variables:

diagnostics:
  example:
    description:
    variables:
      ta:
        mip: Amon
        preprocessor: pp
        timerange: 2000/2002
        reference_dataset: ERA5
        additional_datasets:
          - dataset: CanESM5
            project: CMIP6
            exp: historical
            ensemble: r1i1p1f1
            grid: gn
      pr:
    scripts:
      plot:
        script: examples/diagnostic.py
        ancestors: [example/ta]
        quickplot:
          plot_type: pcolormesh
"""


def test_recipe_is_valid(schema: dict[str, Any]) -> None:
    recipe = yaml.safe_load(RECIPE)
    jsonschema.validate(recipe, schema)


def test_validate_recipe_valid() -> None:
    assert validate_recipe(yaml.safe_load(RECIPE)) == []


@pytest.mark.parametrize("step", esmvalcore.preprocessor.INITIAL_STEPS)
def test_validate_recipe_initial_step(step: str) -> None:
    """Check that the initial steps cannot be used in a recipe."""
    recipe = yaml.safe_load(RECIPE)
    recipe["preprocessors"]["pp"][step] = False
    issues = [str(issue) for issue in validate_recipe(recipe)]
    message = (
        f"preprocessors > pp > {step}: this preprocessor function cannot be "
        "used in a recipe, because it is always applied when the data is "
        "loaded."
    )
    assert issues == [message]


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
        (
            ["documentation", "project"],
            ["c3s-magic"],
            "documentation > project: unknown key. Did you mean 'projects'?",
        ),
        (
            ["diagnostics", "example", "plot_types"],
            ["zonal"],
            "diagnostics > example > plot_types: unknown key.",
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
    assert [str(i) for i in validate_recipe(recipe)] == [message]


def test_validate_recipe_not_a_mapping() -> None:
    assert validate_recipe(None) == [
        RecipeIssue((), "expected a mapping, got None."),
    ]


def test_validate_recipe_multiple_problems() -> None:
    recipe = yaml.safe_load(RECIPE)
    recipe["preprocessors"]["pp"]["regird"] = {}
    recipe["diagnostics"]["example"].pop("scripts")
    assert [str(i) for i in validate_recipe(recipe)] == [
        (
            "preprocessors > pp > regird: unknown preprocessor function. "
            "Did you mean 'regrid'?"
        ),
        "diagnostics > example: missing required key 'scripts'.",
    ]


INVALID_RECIPE = """
documentation: {title: Test, description: Test., authors: [andela_bouwe]}
datasets:
  - {dataset: CanESM2, project: CMIP5, exp: historical, ensemble: r1i1p1}
  - {project: CMIP5, tier: 7, exp: historical, ensemble: r1i1p1}
preprocessors:
  base: &base
    area_statistics:
      operator: 1
  derived:
    <<: *base
    convert_units:
diagnostics:
  example:
    variables: {ta: {preprocessor: derived}}
"""


def test_validate_recipe_positions() -> None:
    recipe = yaml.safe_load(INVALID_RECIPE)
    issues = validate_recipe(recipe, INVALID_RECIPE)
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
    issues = [
        issue
        for issue in validate_recipe(recipe, MERGE_RECIPE)
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
        for issue in validate_recipe(
            yaml.safe_load(MERGE_RECIPE),
            MERGE_RECIPE,
        )
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
    issues = validate_recipe(
        yaml.safe_load(SELF_MERGE_RECIPE),
        SELF_MERGE_RECIPE,
    )
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
        for issue in validate_recipe(yaml.safe_load(text), text)
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
    issues = validate_recipe(yaml.safe_load(text), text)
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
    issues = validate_recipe(yaml.safe_load(ALIAS_RECIPE), ALIAS_RECIPE)
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


def test_validate_recipe_positions_not_found() -> None:
    recipe = yaml.safe_load(INVALID_RECIPE)
    issues = validate_recipe(recipe, "documentation: {}\n")
    assert issues
    assert all(i.line is None and i.column is None for i in issues)


def test_validate_recipe_positions_invalid_text() -> None:
    recipe = yaml.safe_load(INVALID_RECIPE)
    issues = validate_recipe(recipe, "documentation: [\n")
    assert issues == validate_recipe(recipe)


def test_validate_recipe_does_not_modify_recipe() -> None:
    recipe = yaml.safe_load(RECIPE)
    recipe["preprocessors"]["pp"]["regrid"] = None
    validate_recipe(recipe)
    assert recipe["preprocessors"]["pp"]["regrid"] is None


def test_recipe_with_schema(tmp_path: Path) -> None:
    recipe = yaml.safe_load(INVALID_RECIPE)
    path = tmp_path / "recipe_test.yml"
    path.write_text(INVALID_RECIPE, encoding="utf-8")
    msg = (
        f"Recipe {path} is invalid:\n"
        "- line 5: datasets > 1: missing required key 'dataset'.\n"
        "- line 5: datasets > 1 > tier: expected at most 3, got 7.\n"
    )
    with pytest.raises(RecipeError, match=re.escape(msg)):
        check.recipe_with_schema(recipe, path)


def test_recipe_with_schema_no_file(tmp_path: Path) -> None:
    recipe = yaml.safe_load(RECIPE)
    recipe["diagnostics"]["example"].pop("scripts")
    path = tmp_path / "recipe_test.yml"
    msg = (
        f"Recipe {path} is invalid:\n"
        "- diagnostics > example: missing required key 'scripts'."
    )
    with pytest.raises(RecipeError, match=re.escape(msg)):
        check.recipe_with_schema(recipe, path)


AUTHORS = [f"author_{i}" for i in range(12)]


@pytest.fixture
def esmvaltool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Diagnostics:
    """Pretend that ESMValTool is installed."""
    diagnostics = Diagnostics(tmp_path / "esmvaltool")
    diagnostics.references.mkdir(parents=True)
    (diagnostics.references / "eyring16gmd.bibtex").write_text("")
    tags = {
        "authors": {a: {"name": a} for a in AUTHORS},
        "projects": {"c3s-magic": "C3S MAGIC project"},
        "realms": {"atmos": "atmosphere", "ocean": "ocean"},
        "themes": {"phys": "physics"},
    }
    diagnostics.tags_config.write_text(
        yaml.safe_dump(tags, sort_keys=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(json_schema, "DIAGNOSTICS", diagnostics)
    monkeypatch.setattr(json_schema, "TAGS", diagnostics.load_tags())
    return diagnostics


def test_schema_without_esmvaltool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(json_schema, "DIAGNOSTICS", Diagnostics(tmp_path))
    monkeypatch.setattr(json_schema, "TAGS", TagsManager())
    schema = generate_recipe_schema()
    documentation = schema["definitions"]["documentation"]["properties"]
    for key in ("authors", "maintainer", "projects", "references"):
        assert documentation[key]["items"] == {"type": "string"}


@pytest.mark.usefixtures("esmvaltool")
def test_schema_with_esmvaltool() -> None:
    schema = generate_recipe_schema()
    documentation = schema["definitions"]["documentation"]["properties"]
    assert documentation["authors"]["items"]["enum"] == AUTHORS
    assert documentation["maintainer"]["items"]["enum"] == AUTHORS
    assert documentation["projects"]["items"]["enum"] == ["c3s-magic"]
    assert documentation["references"]["items"]["enum"] == ["eyring16gmd"]
    diagnostic = schema["definitions"]["diagnostic"]["properties"]
    assert diagnostic["realms"]["items"]["enum"] == ["atmos", "ocean"]
    assert diagnostic["themes"]["items"]["enum"] == ["phys"]


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (
            ("documentation", "authors", 0),
            "author_100",
            (
                "unknown value 'author_100'. Not defined in section 'authors' "
                "of `{esmvaltool}/config-references.yml`, so the provenance "
                "will be incomplete. Did you mean 'author_10'?"
            ),
        ),
        (
            ("documentation", "references", 0),
            "eyring16",
            (
                "unknown value 'eyring16'. No bibtex file in "
                "`{esmvaltool}/references`, so the citation will be "
                "incomplete. Did you mean 'eyring16gmd'?"
            ),
        ),
        (
            ("diagnostics", "example", "realms", 0),
            "atmosphere",
            (
                "unknown value 'atmosphere'. Not defined in section 'realms' "
                "of `{esmvaltool}/config-references.yml`, so the provenance "
                "will be incomplete. Did you mean 'atmos'?"
            ),
        ),
    ],
)
def test_validate_recipe_names(
    esmvaltool: Diagnostics,
    path: tuple[str | int, ...],
    value: str,
    message: str,
) -> None:
    recipe = yaml.safe_load(RECIPE)
    recipe["documentation"]["authors"] = ["author_0"]
    recipe["documentation"]["references"] = ["eyring16gmd"]
    recipe["diagnostics"]["example"]["realms"] = ["atmos"]
    node = recipe
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    expected = message.format(esmvaltool=esmvaltool.path)
    assert validate_recipe(recipe) == [
        RecipeIssue(path, expected, severity="warning"),
    ]


@pytest.mark.usefixtures("esmvaltool")
def test_validate_recipe_names_wrong_type() -> None:
    recipe = yaml.safe_load(RECIPE)
    recipe["documentation"]["authors"] = [1]
    issues = validate_recipe(recipe)
    assert (
        RecipeIssue(
            ("documentation", "authors", 0),
            "expected a string, got 1.",
        )
        in issues
    )


@pytest.mark.usefixtures("esmvaltool")
def test_recipe_with_schema_warnings(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING)
    recipe = yaml.safe_load(RECIPE)
    recipe["documentation"]["authors"] = ["author_100"]
    path = tmp_path / "recipe_test.yml"
    path.write_text(yaml.safe_dump(recipe), encoding="utf-8")
    check.recipe_with_schema(recipe, path)
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    message = caplog.records[0].getMessage()
    assert message.startswith(f"Recipe {path}: line ")
    assert (
        "documentation > authors > 0: unknown value 'author_100'." in message
    )


@pytest.mark.usefixtures("esmvaltool")
def test_validate_recipe_names_updated() -> None:
    recipe = yaml.safe_load(RECIPE)
    recipe["documentation"]["authors"] = ["new_author"]
    assert validate_recipe(recipe)
    json_schema.TAGS.set_tag_value("authors", "new_author", {})
    assert not validate_recipe(recipe)


@pytest.mark.parametrize(
    ("path", "value", "message", "severity"),
    [
        (
            ("datasets", 0, "project"),
            "CMIP8",
            (
                "unknown value 'CMIP8'. Not defined in the `projects` "
                "section of the configuration. Did you mean 'CMIP7'?"
            ),
            "error",
        ),
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
        (
            ("diagnostics", "example", "variables", "ta", "mip"),
            "Amonn",
            (
                "unknown value 'Amonn'. Not a CMOR table of any project. "
                "Did you mean 'Amon'?"
            ),
            "warning",
        ),
    ],
)
def test_validate_recipe_projects_and_mips(
    path: tuple[str | int, ...],
    value: str,
    message: str,
    severity: Literal["error", "warning"],
) -> None:
    recipe = yaml.safe_load(RECIPE)
    node = recipe
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    assert validate_recipe(recipe) == [
        RecipeIssue(path, message, severity=severity),
    ]


@pytest.mark.parametrize("mip", ["*", "A*", "Amo?", "[AO]mon"])
def test_validate_recipe_mip_pattern(mip: str) -> None:
    recipe = yaml.safe_load(RECIPE)
    recipe["datasets"][0]["mip"] = mip
    recipe["diagnostics"]["example"]["variables"]["ta"]["mip"] = mip
    assert validate_recipe(recipe) == []


@pytest.mark.parametrize(
    ("mip", "branding_suffix"),
    [
        ("atmos", "tavg-h2m-hxy-u"),
        ("atmos", "tavg-*"),
        ("atmos", ["tavg-h2m-hxy-u", "tmax-h2m-hxy-u"]),
        ("ocean", "tavg-u-hxy-sea"),
        ("*", "anything"),
    ],
)
def test_validate_recipe_branding_suffix(
    mip: str,
    branding_suffix: str | list[str],
) -> None:
    recipe = yaml.safe_load(RECIPE)
    variable = recipe["diagnostics"]["example"]["variables"]["ta"]
    variable["mip"] = mip
    variable["branding_suffix"] = branding_suffix
    assert validate_recipe(recipe) == []


def test_validate_recipe_branding_suffix_invalid() -> None:
    recipe = yaml.safe_load(RECIPE)
    variable = recipe["diagnostics"]["example"]["variables"]["ta"]
    variable["mip"] = "atmos"
    variable["branding_suffix"] = "tavg-h2m-hxy-uu"
    path = ("diagnostics", "example", "variables", "ta", "branding_suffix")
    assert validate_recipe(recipe) == [
        RecipeIssue(
            path,
            "unknown value 'tavg-h2m-hxy-uu'. Not a branding suffix of a "
            "variable in CMOR table 'atmos'. Did you mean 'tavg-h2m-hxy-u'?",
        ),
    ]


def test_branding_suffix_rules_not_strict() -> None:
    tables: dict[str, json_schema._Tables | None] = {
        "A": json_schema._Tables(
            ("atmos",),
            strict=True,
            branding_suffixes=(("atmos", ("tavg-u-hxy-u",)),),
        ),
        "B": json_schema._Tables(
            ("atmos",),
            strict=False,
            branding_suffixes=(("atmos", ("tpt-u-hxy-u",)),),
        ),
    }
    (rule,) = json_schema._branding_suffix_rules(tables)
    schema = rule["then"]["properties"]["branding_suffix"]
    assert schema["enum"] == ["tavg-u-hxy-u", "tpt-u-hxy-u"]
    assert schema[json_schema.SEVERITY_KEYWORD] == "warning"


def test_validate_recipe_mip_of_other_project() -> None:
    recipe = yaml.safe_load(RECIPE)
    recipe["datasets"][2]["mip"] = "atmos"
    recipe["datasets"][2]["project"] = "CMIP7"
    recipe["datasets"][2].update(
        branding_suffix="*",
        frequency="mon",
        region="glb",
    )
    assert validate_recipe(recipe) == []


@pytest.mark.parametrize(
    ("tables", "severity"),
    [
        (
            {"A": json_schema._Tables(("Amon",), strict=True)},
            "error",
        ),
        (
            {
                "A": json_schema._Tables(("Amon",), strict=True),
                "B": json_schema._Tables(("Omon",), strict=False),
            },
            "warning",
        ),
        ({"A": json_schema._Tables(("Amon",), strict=True), "B": None}, None),
    ],
)
def test_project_rules_without_project(
    tables: dict[str, json_schema._Tables | None],
    severity: str | None,
) -> None:
    rules = json_schema._project_rules(tables)
    without_project = [
        rule
        for rule in rules
        if rule["if"].get("not") == {"required": ["project"]}
    ]
    if severity is None:
        # Any MIP can be used with a project without CMOR tables.
        assert without_project == []
    else:
        (rule,) = without_project
        mip = rule["then"]["properties"]["mip"]
        assert mip["enum"] == sorted(
            name for t in tables.values() if t for name in t.names
        )
        assert mip.get(json_schema.SEVERITY_KEYWORD, "error") == severity


DUPLICATE_RECIPE = """
documentation: {title: Test, description: Test., authors: [andela_bouwe]}
preprocessors:
  base: &base
    area_statistics:
      operator: mean
    area_statistics:
      operator: max
  pp:
    <<: *base
    convert_units:
      units: K
diagnostics:
  example:
    variables:
      tas:
        mip: Amon
      tas:
        mip: day
    scripts: null
"""


def test_validate_recipe_duplicate_keys() -> None:
    recipe = yaml.safe_load(DUPLICATE_RECIPE)
    issues = validate_recipe(recipe, DUPLICATE_RECIPE)
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
    issues = validate_recipe(yaml.safe_load(MERGE_RECIPE), MERGE_RECIPE)
    assert not [i for i in issues if "duplicate key" in i.message]


def test_recipe_with_schema_duplicate_keys(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING)
    recipe = yaml.safe_load(DUPLICATE_RECIPE)
    path = tmp_path / "recipe_test.yml"
    path.write_text(DUPLICATE_RECIPE, encoding="utf-8")
    check.recipe_with_schema(recipe, path)
    assert [r.levelname for r in caplog.records] == ["WARNING", "WARNING"]
    assert caplog.records[1].getMessage() == (
        f"Recipe {path}: line 16: diagnostics > example > variables > tas: "
        "duplicate key, this definition is ignored because 'tas' is defined "
        "again on line 18."
    )


def test_validate_recipe_unknown_preprocessor() -> None:
    recipe = yaml.safe_load(RECIPE)
    recipe["diagnostics"]["example"]["variables"]["ta"]["preprocessor"] = "p"
    assert validate_recipe(recipe) == [
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
    assert validate_recipe(recipe) == [
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
    assert validate_recipe(recipe) == []


@pytest.mark.parametrize("project", ["OBS", "OBS6"])
def test_validate_recipe_missing_facets(project: str) -> None:
    recipe = yaml.safe_load(RECIPE)
    # The dataset is used by both variable groups, but reported once.
    recipe["datasets"][1] = {"dataset": "ERA5", "project": project}
    assert validate_recipe(recipe) == [
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
    assert validate_recipe(recipe) == [
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
    assert validate_recipe(recipe) == []


def test_validate_recipe_unknown_type() -> None:
    recipe = yaml.safe_load(RECIPE)
    recipe["datasets"][1].update(project="OBS6", type="reanalysis")
    assert validate_recipe(recipe) == [
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
    assert validate_recipe(recipe) == [
        RecipeIssue(
            ("diagnostics", "example", "variables", "ta", "type"),
            "unknown value 'atmos'. Not a type of project 'OBS', choose "
            "from 'sat', 'reanaly', 'ground', 'clim', or 'campaign'.",
            severity="warning",
        ),
    ]


@pytest.mark.parametrize(
    ("facet", "text", "anchor"),
    [
        ("tier", "3 for restricted", "#tiers"),
        ("type", "`reanaly`", "#observation-types"),
    ],
)
def test_obs_facet_documentation(
    schema: dict[str, Any],
    facet: str,
    text: str,
    anchor: str,
) -> None:
    facets = schema["definitions"]["facets"]["properties"]
    description = facets[facet]["markdownDescription"]
    assert text in description
    assert f"(https://docs.esmvaltool.org/en/latest/input.html{anchor})" in (
        description
    )


def test_validate_recipe_missing_facets_unknown_project() -> None:
    recipe = yaml.safe_load(RECIPE)
    recipe["datasets"][1]["project"] = ["OBS"]
    assert not [
        i for i in validate_recipe(recipe) if "Needed to find" in i.message
    ]


def test_validate_recipe_missing_facets_esgf_project() -> None:
    recipe = yaml.safe_load(RECIPE)
    del recipe["datasets"][2]["grid"]
    assert validate_recipe(recipe) == [
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
    assert validate_recipe(recipe) == [
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
    issues = [
        i for i in validate_recipe(recipe) if "Needed to find" in i.message
    ]
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
    assert validate_recipe(recipe) == []


@pytest.mark.parametrize("key", ["reference_dataset", "alternative_dataset"])
def test_validate_recipe_unknown_reference_dataset(key: str) -> None:
    recipe = yaml.safe_load(RECIPE)
    recipe["diagnostics"]["example"]["variables"]["ta"][key] = "ERA5|ERA4"
    assert validate_recipe(recipe) == [
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
    variables["pr"] = {"reference_dataset": "CanESM5"}
    assert [i.path[-2:] for i in validate_recipe(recipe)] == [
        ("pr", "reference_dataset"),
    ]


def test_validate_recipe_reference_dataset_pattern() -> None:
    recipe = yaml.safe_load(RECIPE)
    recipe["datasets"][0]["dataset"] = "*"
    recipe["diagnostics"]["example"]["variables"]["ta"][
        "reference_dataset"
    ] = "X"
    assert validate_recipe(recipe) == []


@pytest.mark.parametrize(
    "ancestors",
    [["ta"], ["example/ta", "example/pr"], ["example/*"], ["*/p?"]],
)
def test_validate_recipe_ancestors(ancestors: list[str]) -> None:
    recipe = yaml.safe_load(RECIPE)
    recipe["diagnostics"]["example"]["scripts"]["plot"]["ancestors"] = (
        ancestors
    )
    assert validate_recipe(recipe) == []


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
    assert validate_recipe(recipe) == [
        RecipeIssue(
            ("diagnostics", "example", "scripts", "plot", "ancestors", 1),
            message,
        ),
    ]
