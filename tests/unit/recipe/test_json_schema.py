"""Tests for :mod:`esmvalcore._recipe.json_schema`."""

from __future__ import annotations

import json
import re
import typing
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol

import jsonschema
import pytest
import yaml
from iris.cube import Cube

import esmvalcore.preprocessor
from esmvalcore._recipe import check, json_schema
from esmvalcore._recipe.json_schema import (
    RecipeIssue,
    _annotation_to_schema,
    _clean_rst,
    _parse_docstring,
    _resolve_annotation,
    generate_recipe_schema,
    validate_recipe,
    write_recipe_schema,
)
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
            "Use esmvalcore.preprocessor.regrid for `details`.\n\n"
            "Second paragraph."
        ),
        "kwargs": "Other arguments.",
    }


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


@pytest.fixture(scope="module")
def schema() -> dict[str, Any]:
    return generate_recipe_schema()


def test_schema_is_valid(schema: dict[str, Any]) -> None:
    jsonschema.Draft7Validator.check_schema(schema)


def test_schema_has_all_preprocessor_steps(schema: dict[str, Any]) -> None:
    steps = schema["definitions"]["steps"]
    assert list(steps) == list(esmvalcore.preprocessor.DEFAULT_ORDER)
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
  - {dataset: ACCESS-CM2, project: CMIP6, grid: gn, activity: [CMIP, DCPP]}

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
          - {dataset: CanESM5, project: CMIP6, exp: historical}
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
  - {dataset: CanESM2, project: CMIP5}
  - {project: CMIP5, tier: 7}
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
        # Problems in an anchored mapping are found at the anchor.
        RecipeIssue(
            ("preprocessors", "base", "area_statistics", "operator"),
            "expected a string, got 1.",
            line=9,
            column=7,
        ),
        RecipeIssue(
            ("preprocessors", "derived", "area_statistics", "operator"),
            "expected a string, got 1.",
            line=9,
            column=7,
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
    ("preprocessor", "line"),
    [
        # In a sequence of mappings to merge, the first one has priority.
        ("sequence", 3),
        # With repeated merge keys, the last one has priority.
        ("repeated", 5),
    ],
)
def test_validate_recipe_positions_merge(preprocessor: str, line: int) -> None:
    recipe = yaml.safe_load(MERGE_RECIPE)
    expected = "linear" if line == 3 else "nearest"
    assert (
        recipe["preprocessors"][preprocessor]["regrid"]["scheme"] == expected
    )
    recipe["preprocessors"][preprocessor]["regrid"]["scheme"] = "invalid"
    issues = [
        issue
        for issue in validate_recipe(recipe, MERGE_RECIPE)
        if issue.path[:2] == ("preprocessors", preprocessor)
    ]
    assert len(issues) == 1
    assert issues[0].path[-1] == "scheme"
    assert issues[0].line == line


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


def test_write_recipe_schema(tmp_path: Path) -> None:
    path = tmp_path / "recipe_schema.json"
    write_recipe_schema(path)
    assert json.loads(path.read_text(encoding="utf-8")) == (
        generate_recipe_schema()
    )


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
