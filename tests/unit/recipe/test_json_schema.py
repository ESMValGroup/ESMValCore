"""Tests for :mod:`esmvalcore._recipe.json_schema`."""

from __future__ import annotations

import inspect
import re
import typing
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

import jsonschema
import pytest
import yaml
from iris.cube import Cube

import esmvalcore.io.esgf.facets
import esmvalcore.preprocessor
from esmvalcore._recipe import json_schema
from esmvalcore._recipe.json_schema import (
    _annotation_to_schema,
    _clean_rst,
    _parse_docstring,
    _resolve_annotation,
    _rst_to_text,
    generate_recipe_schema,
)
from esmvalcore._recipe.validate import RecipeIssue, validate_recipe
from esmvalcore.config._diagnostics import Diagnostics, TagsManager
from tests.unit.recipe.example_recipes import AUTHORS, RECIPE

if TYPE_CHECKING:
    from esmvalcore.config import Session

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


def test_recipe_is_valid(schema: dict[str, Any]) -> None:
    recipe = yaml.safe_load(RECIPE)
    jsonschema.validate(recipe, schema)


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


def test_schema_with_session(session: Session) -> None:
    session["projects"] = {"CMIP6": session["projects"]["CMIP6"]}
    schema = generate_recipe_schema(session)
    project = schema["definitions"]["facets"]["properties"]["project"]
    assert project["enum"] == ["CMIP6"]


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
def test_validate_recipe_names_updated() -> None:
    recipe = yaml.safe_load(RECIPE)
    recipe["documentation"]["authors"] = ["new_author"]
    assert validate_recipe(recipe)
    json_schema.TAGS.set_tag_value("authors", "new_author", {})
    assert not validate_recipe(recipe)


def test_validate_recipe_unknown_project() -> None:
    recipe = yaml.safe_load(RECIPE)
    recipe["datasets"][0]["project"] = "CMIP8"
    assert validate_recipe(recipe) == [
        RecipeIssue(
            ("datasets", 0, "project"),
            (
                "unknown value 'CMIP8'. Not defined in the `projects` "
                "section of the configuration. Did you mean 'CMIP7'?"
            ),
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
