"""JSON Schema for recipes.

The schema describes the structure of a recipe and the arguments of all
preprocessor functions. ESMValCore uses it to validate recipes, and editors can
use it to offer validation, completion, and documentation while writing a
recipe. The preprocessor part is generated from the signatures and docstrings
of the preprocessor functions.
"""

from __future__ import annotations

import ast
import builtins
import collections.abc
import contextlib
import dataclasses
import difflib
import functools
import importlib
import inspect
import json
import re
import sys
import types
import typing
from pathlib import Path
from typing import Any, Literal

import jsonschema
import yaml

import esmvalcore.preprocessor
from esmvalcore import __version__
from esmvalcore.config._diagnostics import DIAGNOSTICS, TAGS
from esmvalcore.preprocessor import (
    DEFAULT_ORDER,
    FINAL_STEPS,
    INITIAL_STEPS,
    MULTI_MODEL_FUNCTIONS,
)

JSONSchema = dict[str, Any]

SCHEMA_DIALECT = "http://json-schema.org/draft-07/schema#"

# Schema keyword for the severity of a value that is not in an ``enum``.
SEVERITY_KEYWORD = "x-severity"

# Preprocessor steps that support excluding datasets with ``exclude``.
EXCLUDE_STEPS = MULTI_MODEL_FUNCTIONS | {"weighting_landsea_fraction"}

# Arguments that ESMValCore fills in from the dataset if they are missing.
AUTOMATIC_ARGUMENTS = {
    "align_metadata": {"target_mip", "target_short_name"},
    "regrid_time": {"frequency"},
}

# Arguments that ESMValCore always sets, so they cannot be used in a recipe.
# For the initial and final steps, this also includes all arguments without
# a default value.
INTERNAL_ARGUMENTS: dict[str, set[str]] = {
    step: {"session"} for step in INITIAL_STEPS + FINAL_STEPS
}
INTERNAL_ARGUMENTS["bias"] = {"reference"}
INTERNAL_ARGUMENTS["derive"] = {
    "long_name",
    "short_name",
    "standard_name",
    "units",
}
INTERNAL_ARGUMENTS["distance_metric"] = {"reference"}
INTERNAL_ARGUMENTS["ensemble_statistics"] = {"output_products"}
INTERNAL_ARGUMENTS["multi_model_statistics"] = {"output_products"}
INTERNAL_ARGUMENTS["save"].add("compute")

# Annotation types that map directly to a JSON Schema type.
SIMPLE_TYPES: dict[Any, JSONSchema] = {
    type(None): {"type": "null"},
    bool: {"type": "boolean"},
    int: {"type": "integer"},
    float: {"type": "number"},
    str: {"type": "string"},
    Path: {"type": "string"},
}

ARRAY_TYPES = (
    list,
    tuple,
    set,
    frozenset,
    collections.abc.Collection,
    collections.abc.Iterable,
    collections.abc.Sequence,
    collections.abc.Set,
)

OBJECT_TYPES = (dict, collections.abc.Mapping)

# Python types of the values that YAML can represent.
YAML_TYPES = (bool, int, float, str, list, tuple, dict, type(None))


class _UnknownMeta(type):
    """Metaclass that absorbs any operation used in an annotation."""

    def __getattr__(cls, name: str) -> type[_Unknown]:
        return _Unknown

    def __getitem__(cls, item: object) -> type[_Unknown]:
        return _Unknown

    def __or__(cls, other: object) -> type[_Unknown]:
        return _Unknown

    def __ror__(cls, other: object) -> type[_Unknown]:
        return _Unknown


class _Unknown(metaclass=_UnknownMeta):
    """Placeholder for a name in an annotation that cannot be resolved."""


@functools.cache
def _type_checking_imports(module_name: str) -> dict[str, Any]:
    """Import the names that a module only imports for type checking."""
    module = sys.modules[module_name]
    try:
        tree = ast.parse(inspect.getsource(module))
    except (OSError, TypeError):
        return {}
    names = {}
    for node in tree.body:
        if not (
            isinstance(node, ast.If)
            and ast.unparse(node.test).endswith("TYPE_CHECKING")
        ):
            continue
        for stmt in node.body:
            try:
                if isinstance(stmt, ast.ImportFrom):
                    package = importlib.import_module(
                        "." * stmt.level + (stmt.module or ""),
                        module.__package__,
                    )
                    for alias in stmt.names:
                        names[alias.asname or alias.name] = getattr(
                            package,
                            alias.name,
                        )
                elif isinstance(stmt, ast.Import):
                    for alias in stmt.names:
                        top_level = alias.name.split(".")[0]
                        imported = importlib.import_module(alias.name)
                        names[alias.asname or top_level] = (
                            imported
                            if alias.asname
                            else importlib.import_module(top_level)
                        )
            except (ImportError, AttributeError):
                continue
    return names


class _AnnotationNamespace(dict):
    """Namespace for evaluating annotations that are stored as strings.

    Names are looked up in the module that defines the function, then in the
    names it imports for type checking only, the builtins,
    :mod:`collections.abc`, and :mod:`typing`. Names that cannot be found
    become :class:`_Unknown`.
    """

    def __init__(self, module_name: str) -> None:
        super().__init__()
        self._module_name = module_name

    def __missing__(self, name: str) -> object:
        for namespace in (
            vars(sys.modules[self._module_name]),
            _type_checking_imports(self._module_name),
            vars(builtins),
            vars(collections.abc),
            vars(typing),
        ):
            if name in namespace:
                return namespace[name]
        return _Unknown


def _resolve_annotation(
    annotation: object,
    function: collections.abc.Callable[..., object],
) -> object:
    """Turn an annotation stored as a string into a type.

    Because ESMValCore uses ``from __future__ import annotations``, all
    annotations are stored as strings. They cannot be evaluated with
    :func:`typing.get_type_hints` or ``inspect.signature(eval_str=True)``,
    because many refer to names that are only imported for type checking.
    """
    if not isinstance(annotation, str):
        return annotation
    namespace = _AnnotationNamespace(inspect.unwrap(function).__module__)
    try:
        return eval(annotation, {}, namespace)  # noqa: S307
    except (AttributeError, NameError, SyntaxError, TypeError):
        return _Unknown


def _is_python_only(annotation: object) -> bool:
    """Check if values of a type cannot be written in YAML.

    This is the case for classes such as :class:`iris.cube.Cube`, but not
    for protocols, because YAML values may implement them.
    """
    return (
        isinstance(annotation, type)
        and annotation not in (_Unknown, Any)
        and annotation not in SIMPLE_TYPES
        and annotation not in ARRAY_TYPES + OBJECT_TYPES
        and not issubclass(annotation, YAML_TYPES)
        and not getattr(annotation, "_is_protocol", False)
    )


def _union_schema(args: tuple[object, ...]) -> JSONSchema | None:
    """Convert the options of a union type to a JSON Schema."""
    options = [
        _annotation_to_schema(arg) for arg in args if not _is_python_only(arg)
    ]
    if not options or any(option is None for option in options):
        return None
    options = [o for i, o in enumerate(options) if o not in options[:i]]
    if len(options) == 1:
        return options[0]
    return {"anyOf": options}


def _array_schema(origin: object, args: tuple[object, ...]) -> JSONSchema:
    """Convert a sequence type to a JSON Schema."""
    schema: JSONSchema = {"type": "array"}
    if origin is tuple and args and args[-1] is not Ellipsis:
        items = [_annotation_to_schema(arg) for arg in args]
        schema["items"] = [{} if i is None else i for i in items]
        schema["minItems"] = schema["maxItems"] = len(args)
    elif args:
        item = _annotation_to_schema(args[0])
        if item is not None:
            schema["items"] = item
    return schema


def _annotation_to_schema(annotation: object) -> JSONSchema | None:
    """Convert a type annotation to a JSON Schema.

    Returns :obj:`None` if the annotation does not constrain the value, for
    example because it is a type that cannot be written in a recipe. The
    recipe may still provide a value that ESMValCore converts to that type,
    so such values are not rejected. In a union, types that cannot be written
    in a recipe are ignored.
    """
    while isinstance(annotation, typing.TypeAliasType):
        annotation = annotation.__value__
    if annotation is None:
        annotation = type(None)
    if annotation in SIMPLE_TYPES:
        return dict(SIMPLE_TYPES[annotation])

    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    schema: JSONSchema | None = None
    if origin is Literal:
        schema = {"enum": list(args)}
    elif origin in (typing.Union, types.UnionType):
        schema = _union_schema(args)
    elif annotation in ARRAY_TYPES or origin in ARRAY_TYPES:
        schema = _array_schema(origin, args)
    elif annotation in OBJECT_TYPES or origin in OBJECT_TYPES:
        schema = {"type": "object"}
    return schema


def _to_json_value(value: object) -> object:
    """Convert a default value to JSON, raising TypeError if not possible."""
    if isinstance(value, tuple | list):
        return [_to_json_value(v) for v in value]
    json.dumps(value)
    return value


def _clean_rst(text: str) -> str:
    """Remove reStructuredText markup that does not read well in an editor."""
    # Roles with an explicit target, e.g. :ref:`text <target>`, are replaced
    # by their text, other roles, e.g. :math:`< 1`, by their content.
    text = re.sub(r":[\w:]+:`~?([^`<]+?)\s+<[^`>]+>`", r"\1", text)
    text = re.sub(r":[\w:]+:`~?([^`]*)`", r"\1", text)
    return text.replace("``", "`")


def _parse_docstring(docstring: str) -> tuple[str, dict[str, str]]:
    """Get the summary and parameter descriptions from a numpy docstring."""
    lines = docstring.splitlines()
    summary = []
    for line in lines:
        if not line.strip():
            break
        summary.append(line.strip())

    parameters: dict[str, list[str]] = {}
    in_parameters = False
    name = None
    for i, line in enumerate(lines):
        next_line = lines[i + 1] if i + 1 < len(lines) else ""
        if next_line.startswith("---") and line.strip():
            in_parameters = line.strip() == "Parameters"
            name = None
            continue
        if not in_parameters or line.startswith("---"):
            continue
        if match := re.match(r"^\*{0,2}(\w+)\s*:", line):
            name = match.group(1)
            parameters[name] = []
        elif name is not None:
            parameters[name].append(line.strip())

    descriptions = {
        name: _clean_rst(
            re.sub(r"(?<!\n)\n(?!\n)", " ", "\n".join(text).strip()),
        )
        for name, text in parameters.items()
    }
    return _clean_rst(" ".join(summary)), descriptions


def _parameter_schema(
    param: inspect.Parameter,
    function: collections.abc.Callable[..., object],
    description: str | None,
) -> JSONSchema:
    """Create a JSON Schema for an argument of a preprocessor function."""
    annotation = _resolve_annotation(param.annotation, function)
    schema = _annotation_to_schema(annotation) or {}
    if description:
        schema["description"] = description
    if param.default is not inspect.Parameter.empty:
        with contextlib.suppress(TypeError):
            schema["default"] = _to_json_value(param.default)
    return schema


def _preprocessor_step_schema(step: str) -> JSONSchema:
    """Create a JSON Schema for the settings of a preprocessor step."""
    function = getattr(esmvalcore.preprocessor, step)
    summary, descriptions = _parse_docstring(inspect.getdoc(function) or "")
    parameters = list(inspect.signature(function).parameters.values())[1:]

    always_applied = step in INITIAL_STEPS + FINAL_STEPS
    internal = INTERNAL_ARGUMENTS.get(step, set())
    automatic = AUTOMATIC_ARGUMENTS.get(step, set())

    properties: JSONSchema = {}
    required = []
    additional_properties = False
    for param in parameters:
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            additional_properties = True
            continue
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            continue
        no_default = param.default is inspect.Parameter.empty
        if param.name in internal or (always_applied and no_default):
            continue
        if no_default and param.name not in automatic:
            required.append(param.name)
        properties[param.name] = _parameter_schema(
            param,
            function,
            descriptions.get(param.name),
        )

    if step in EXCLUDE_STEPS:
        properties["exclude"] = {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Datasets to exclude from this preprocessor step. Use "
                "`reference_dataset` or `alternative_dataset` to refer to the "
                "dataset defined in the variable."
            ),
        }

    settings: JSONSchema = {
        "type": "object",
        "properties": properties,
        "additionalProperties": additional_properties,
    }
    if required:
        settings["required"] = required

    options: list[JSONSchema] = [
        settings,
        {
            "const": False,
            "description": "Disable this step, if it is enabled by default.",
        },
    ]
    if not required:
        options.append(
            {
                "type": ["null", "boolean"],
                "description": "Enable this step with its default settings.",
            },
        )
    return {"description": summary, "anyOf": options}


def _preprocessor_schema() -> JSONSchema:
    """Create a JSON Schema for a preprocessor."""
    properties: JSONSchema = {
        "custom_order": {
            "type": "boolean",
            "default": False,
            "description": (
                "Apply the preprocessor steps in the order in which they "
                "are listed instead of the default order."
            ),
        },
    }
    for step in DEFAULT_ORDER:
        properties[step] = {"$ref": f"#/definitions/steps/{step}"}
    return {
        "type": "object",
        "description": "Preprocessor steps and their settings.",
        "properties": properties,
        "additionalProperties": False,
    }


def _string_or_list(description: str) -> JSONSchema:
    return {
        "anyOf": [
            {"type": "string"},
            {"type": "array", "items": {"type": "string"}},
        ],
        "description": description,
    }


class _Names(typing.NamedTuple):
    """Names that are defined by the installed ESMValTool."""

    names: tuple[str, ...]
    hint: str
    """Explains what an unknown name means and where to define it."""


def _known_names() -> dict[str, _Names]:
    """Get the names that are defined by the installed ESMValTool.

    These are the tags in ``config-references.yml``, by section, and the
    references that have a bibtex file. Without ESMValTool, there are none.
    """
    names = {
        section: _Names(
            tuple(TAGS[section] or ()),
            f"Not defined in section '{section}' of "
            f"`{DIAGNOSTICS.tags_config}`, so the provenance will be "
            "incomplete.",
        )
        for section in ("authors", "projects", "realms", "themes")
        if section in TAGS
    }
    if DIAGNOSTICS.references.is_dir():
        names["references"] = _Names(
            tuple(
                sorted(
                    p.stem for p in DIAGNOSTICS.references.glob("*.bibtex")
                ),
            ),
            f"No bibtex file in `{DIAGNOSTICS.references}`, so the citation "
            "will be incomplete.",
        )
    return names


def _names_list(
    description: str,
    section: str,
    names: dict[str, _Names],
    min_items: int = 0,
) -> JSONSchema:
    """Create a JSON Schema for a list of tags or references."""
    schema = _string_list(description, min_items)
    if section in names:
        # An unknown name only makes the provenance less complete, so it is
        # reported as a warning. Editors also show it as a warning.
        schema["items"] = {
            "type": "string",
            "enum": list(names[section].names),
            "errorMessage": names[section].hint,
            SEVERITY_KEYWORD: "warning",
        }
    return schema


def _string_list(description: str, min_items: int = 0) -> JSONSchema:
    schema: JSONSchema = {
        "type": "array",
        "items": {"type": "string"},
        "description": description,
    }
    if min_items:
        schema["minItems"] = min_items
    return schema


def _facet_value(description: str) -> JSONSchema:
    """Create a JSON Schema for :obj:`esmvalcore.typing.FacetValue`."""
    return {
        "anyOf": [
            {"type": ["string", "number"]},
            {"type": "array", "items": {"type": "string"}},
        ],
        "description": description,
    }


def _facets_schema() -> JSONSchema:
    """Create a JSON Schema for the facets of a dataset or variable.

    See doc/reference/facets.rst for the facets that ESMValCore uses.
    """
    year = {"type": "integer", "minimum": 0, "maximum": 10000}
    return {
        "type": "object",
        "properties": {
            "dataset": {
                "type": "string",
                "description": "Name of the dataset.",
            },
            "project": {
                "type": "string",
                "description": "Project, for example CMIP6.",
            },
            "short_name": {
                "type": "string",
                "description": "Short name of the variable.",
            },
            "mip": {
                "type": "string",
                "description": "MIP table, for example Amon.",
            },
            "exp": _string_or_list(
                "Experiment, or a list of experiments to concatenate.",
            ),
            "ensemble": _string_or_list("Ensemble member(s)."),
            "activity": _facet_value(
                "Activity, for example CMIP or ScenarioMIP.",
            ),
            "branding_suffix": _facet_value(
                "Variable branding suffix (CMIP7).",
            ),
            "domain": _facet_value("CORDEX domain, for example EUR-11."),
            "driver": _facet_value(
                "Driving global model of a CORDEX dataset.",
            ),
            "frequency": _facet_value("Frequency, for example mon."),
            "grid": _facet_value("Grid label, for example gn."),
            "institute": _facet_value("Institute that produced the data."),
            "product": _facet_value("Product (CMIP5)."),
            "rcm_version": _facet_value(
                "Version of the regional climate model (CORDEX).",
            ),
            "realm": _facet_value("Realm, for example atmos."),
            "region": _facet_value("Region (CMIP7)."),
            "version": _facet_value("Version of the dataset."),
            "timerange": {
                "type": ["string", "integer"],
                "description": (
                    "Time range in ISO 8601 format, for example 1990/2000."
                ),
            },
            "start_year": {
                **year,
                "description": "First year. Alternative to `timerange`.",
            },
            "end_year": {
                **year,
                "description": "Last year. Alternative to `timerange`.",
            },
            "alias": {
                "type": "string",
                "description": "Name of the dataset to use in diagnostics.",
            },
            "tier": {
                "type": "integer",
                "minimum": 1,
                "maximum": 3,
                "description": "Tier of an observational dataset.",
            },
            "derive": {
                "type": "boolean",
                "description": "Derive the variable from other variables.",
            },
            "force_derivation": {
                "type": "boolean",
                "description": (
                    "Derive the variable even if it is available directly."
                ),
            },
            "supplementary_variables": {
                "type": "array",
                "items": {
                    "allOf": [{"$ref": "#/definitions/facets"}],
                    "properties": {
                        "skip": {
                            "type": "boolean",
                            "description": (
                                "Do not add this supplementary variable."
                            ),
                        },
                    },
                },
                "description": (
                    "Supplementary variables, for example cell measures."
                ),
            },
        },
    }


def _dataset_schema() -> JSONSchema:
    return {
        "allOf": [{"$ref": "#/definitions/facets"}],
        "required": ["dataset"],
    }


def _variable_schema() -> JSONSchema:
    return {
        "allOf": [{"$ref": "#/definitions/facets"}],
        "properties": {
            "preprocessor": {
                "type": "string",
                "description": "Name of the preprocessor to apply.",
            },
            "reference_dataset": {
                "type": "string",
                "description": "Reference dataset for this variable.",
            },
            "alternative_dataset": {
                "type": "string",
                "description": "Alternative dataset for this variable.",
            },
            "additional_datasets": {
                "type": "array",
                "items": {"$ref": "#/definitions/dataset"},
                "description": "Datasets to use for this variable only.",
            },
        },
    }


def _script_schema() -> JSONSchema:
    return {
        "type": "object",
        "properties": {
            "script": {
                "type": "string",
                "description": (
                    "Path to the diagnostic script, relative to the "
                    "diagnostics directory or absolute."
                ),
            },
            "ancestors": _string_list(
                "Tasks whose output this script uses, as "
                "`diagnostic/variable_group` or `diagnostic/script`. "
                "Wildcards are allowed.",
            ),
        },
        "required": ["script"],
    }


def _diagnostic_schema(names: dict[str, _Names]) -> JSONSchema:
    return {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Title."},
            "description": {
                "type": ["string", "null"],
                "description": "Description of the diagnostic.",
            },
            "themes": _names_list("Themes.", "themes", names),
            "realms": _names_list("Realms.", "realms", names),
            "variables": {
                "type": ["object", "null"],
                "additionalProperties": {
                    "anyOf": [
                        {"$ref": "#/definitions/variable"},
                        {"type": "null"},
                    ],
                },
                "description": "Variable groups to preprocess.",
            },
            "scripts": {
                "type": ["object", "null"],
                "additionalProperties": {"$ref": "#/definitions/script"},
                "description": "Diagnostic scripts to run.",
            },
            "additional_datasets": {
                "type": "array",
                "items": {"$ref": "#/definitions/dataset"},
                "description": "Datasets to use for this diagnostic only.",
            },
        },
        "required": ["scripts"],
        "additionalProperties": False,
    }


def _documentation_schema(names: dict[str, _Names]) -> JSONSchema:
    return {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Recipe title."},
            "description": {
                "type": "string",
                "description": "Description of the recipe.",
            },
            "authors": _names_list(
                "Authors, as defined in `config-references.yml`.",
                "authors",
                names,
                min_items=1,
            ),
            "maintainer": _names_list(
                "Maintainers, as defined in `config-references.yml`.",
                "authors",
                names,
            ),
            "projects": _names_list(
                "Projects, as defined in `config-references.yml`.",
                "projects",
                names,
            ),
            "references": _names_list(
                "References, as bibtex keys.",
                "references",
                names,
            ),
        },
        "required": ["title", "description", "authors"],
        "additionalProperties": False,
    }


def generate_recipe_schema() -> JSONSchema:
    """Generate a JSON Schema for recipes.

    If ESMValTool is installed, the schema only allows the authors, projects,
    realms, and themes defined in its ``config-references.yml`` file and the
    references that have a bibtex file.

    Returns
    -------
    :
        A JSON Schema (draft-07) that describes a recipe.
    """
    return _recipe_schema(_known_names())


def _recipe_schema(names: dict[str, _Names]) -> JSONSchema:
    """Generate a JSON Schema for recipes, given the known tags."""
    return {
        "$schema": SCHEMA_DIALECT,
        "title": "ESMValTool recipe",
        "description": (
            f"Recipe for ESMValTool, generated by ESMValCore {__version__}."
        ),
        "type": "object",
        "properties": {
            "documentation": {"$ref": "#/definitions/documentation"},
            "datasets": {
                "type": "array",
                "items": {"$ref": "#/definitions/dataset"},
                "description": "Datasets to use in all diagnostics.",
            },
            "preprocessors": {
                "type": "object",
                "additionalProperties": {
                    "$ref": "#/definitions/preprocessor",
                },
                "description": "Preprocessors that variables can refer to.",
            },
            "diagnostics": {
                "type": ["object", "null"],
                "additionalProperties": {"$ref": "#/definitions/diagnostic"},
                "description": "Diagnostics to run.",
            },
        },
        "required": ["documentation", "diagnostics"],
        "definitions": {
            "documentation": _documentation_schema(names),
            "facets": _facets_schema(),
            "dataset": _dataset_schema(),
            "variable": _variable_schema(),
            "script": _script_schema(),
            "diagnostic": _diagnostic_schema(names),
            "preprocessor": _preprocessor_schema(),
            "steps": {
                step: _preprocessor_step_schema(step) for step in DEFAULT_ORDER
            },
        },
    }


def write_recipe_schema(path: Path | str) -> None:
    """Write a JSON Schema for recipes to a file.

    Parameters
    ----------
    path:
        Path to the file where the schema will be written.
    """
    Path(path).write_text(
        json.dumps(generate_recipe_schema(), indent=2) + "\n",
        encoding="utf-8",
    )


# Descriptions of JSON Schema types in terms of YAML, for error messages.
TYPE_DESCRIPTIONS = {
    "array": "a list",
    "boolean": "true or false",
    "integer": "an integer",
    "null": "nothing",
    "number": "a number",
    "object": "a mapping",
    "string": "a string",
}


def _get_validator() -> jsonschema.Draft7Validator:
    # The known names are part of the cache key, because they may change,
    # for example when the tags are updated.
    return _create_validator(tuple(_known_names().items()))


@functools.cache
def _create_validator(
    names: tuple[tuple[str, _Names], ...],
) -> jsonschema.Draft7Validator:
    return jsonschema.Draft7Validator(_recipe_schema(dict(names)))


def _normalize(recipe: object) -> object:
    """Replace preprocessor steps without settings by empty settings.

    ESMValCore treats ``step:`` and ``step: true`` as ``step: {}``, so this
    makes validation report missing arguments for such steps.
    """
    if not isinstance(recipe, dict) or not isinstance(
        recipe.get("preprocessors"),
        dict,
    ):
        return recipe
    preprocessors = {
        name: {
            step: {}
            if step != "custom_order" and settings in (None, True)
            else settings
            for step, settings in profile.items()
        }
        if isinstance(profile, dict)
        else profile
        for name, profile in recipe["preprocessors"].items()
    }
    return {**recipe, "preprocessors": preprocessors}


def _json_types(value: object) -> set[str]:
    """Get the JSON Schema types that a value is an instance of."""
    checker = jsonschema.Draft7Validator.TYPE_CHECKER
    return {t for t in TYPE_DESCRIPTIONS if checker.is_type(value, t)}


def _allows_type(schema: JSONSchema, value: object) -> bool:
    """Check if a schema allows values of the type of `value`."""
    types = _json_types(value)
    if "type" in schema:
        allowed = schema["type"]
        allowed = allowed if isinstance(allowed, list) else [allowed]
        return bool(types & set(allowed))
    if "enum" in schema:
        return any(types & _json_types(v) for v in schema["enum"])
    if "const" in schema:
        return bool(types & _json_types(schema["const"]))
    return True


def _expected(schema: JSONSchema) -> list[str] | None:
    """Describe the values that a schema allows, if it is simple enough."""
    if "enum" in schema:
        return ["one of " + ", ".join(repr(v) for v in schema["enum"])]
    if "const" in schema:
        return [repr(schema["const"])]
    if "type" in schema:
        types = schema["type"]
        types = types if isinstance(types, list) else [types]
        return [TYPE_DESCRIPTIONS[t] for t in types]
    return None


def _join_or(items: list[str]) -> str:
    """Join items like "a, b, or c"."""
    if len(items) <= 2:
        return " or ".join(items)
    return ", ".join(items[:-1]) + ", or " + items[-1]


def _relevant_errors(
    error: jsonschema.ValidationError,
    alternatives: list[str] | None = None,
) -> list[tuple[jsonschema.ValidationError, list[str]]]:
    """Get the errors from the option of ``anyOf`` that matches the value.

    Returns pairs of an error and descriptions of the other values that
    would have been accepted.
    """
    if error.validator == "anyOf" and error.context:
        options = [
            i
            for i, option in enumerate(error.validator_value)
            if _allows_type(option, error.instance)
        ]
        if len(options) == 1:
            others = [
                description
                for i, option in enumerate(error.validator_value)
                if i != options[0]
                for description in _expected(option) or []
            ]
            return [
                relevant
                for e in error.context
                if e.relative_schema_path[0] == options[0]
                for relevant in _relevant_errors(
                    e,
                    others if e.absolute_path == error.absolute_path else [],
                )
            ]
    return [(error, alternatives or [])]


def _location(path: collections.abc.Iterable[str | int]) -> str:
    return " > ".join(str(p) for p in path) or "recipe"


@dataclasses.dataclass(frozen=True)
class RecipeIssue:
    """A problem found in a recipe.

    Parameters
    ----------
    path:
        Keys and list indices that lead to the problem in the recipe.
    message:
        Description of the problem.
    line:
        Line number in the recipe file, starting at 1, if known.
    column:
        Column number in the recipe file, starting at 1, if known.
    severity:
        Errors prevent the recipe from running, warnings do not.
    """

    path: tuple[str | int, ...]
    message: str
    line: int | None = None
    column: int | None = None
    severity: Literal["error", "warning"] = "error"

    def __str__(self) -> str:
        text = f"{_location(self.path)}: {self.message}"
        if self.line is not None:
            text = f"line {self.line}: {text}"
        return text


def _suggest(name: str, choices: collections.abc.Iterable[str]) -> str:
    close = difflib.get_close_matches(name, list(choices), n=1)
    return f" Did you mean '{close[0]}'?" if close else ""


def _in_step(path: tuple[str | int, ...]) -> bool:
    """Check if a path points to the settings of a preprocessor step."""
    return len(path) == 3 and path[0] == "preprocessors"


def _describe_unknown_keys(
    error: jsonschema.ValidationError,
) -> list[RecipeIssue]:
    path = tuple(error.absolute_path)
    known = error.schema.get("properties", {})
    if len(path) == 2 and path[0] == "preprocessors":
        problem = "unknown preprocessor function."
    elif _in_step(path):
        problem = "unknown argument."
    else:
        problem = "unknown key."
    issues = []
    for key in error.instance:
        if key in known:
            continue
        message = problem + _suggest(key, known)
        if _in_step(path):
            valid = ", ".join(sorted(known)) or "none"
            message += f" Valid arguments are: {valid}."
        issues.append(RecipeIssue((*path, key), message))
    return issues


def _describe_missing_keys(
    error: jsonschema.ValidationError,
) -> list[RecipeIssue]:
    path = tuple(error.absolute_path)
    missing = [k for k in error.validator_value if k not in error.instance]
    what = "argument" if _in_step(path) else "key"
    if len(missing) > 1:
        what += "s"
    names = ", ".join(repr(k) for k in missing)
    return [RecipeIssue(path, f"missing required {what} {names}.")]


def _describe_invalid_value(
    error: jsonschema.ValidationError,
    alternatives: list[str],
) -> list[RecipeIssue]:
    if hint := error.schema.get("errorMessage"):
        # Listing all choices is not helpful, e.g. for authors.
        message = f"unknown value {error.instance!r}. {hint}"
        message += _suggest(
            str(error.instance),
            map(str, error.validator_value),
        )
        severity = error.schema.get(SEVERITY_KEYWORD, "error")
        return [
            RecipeIssue(
                tuple(error.absolute_path),
                message,
                severity=severity,
            ),
        ]
    choices = [repr(v) for v in error.validator_value]
    if alternatives:
        choices = [
            *choices[:-1],
            f"{choices[-1]}, or {_join_or(alternatives)}",
        ]
    message = (
        f"invalid value {error.instance!r}, choose from {', '.join(choices)}."
    )
    return [RecipeIssue(tuple(error.absolute_path), message)]


def _describe_wrong_type(
    error: jsonschema.ValidationError,
) -> list[RecipeIssue]:
    options = (
        error.validator_value
        if error.validator == "anyOf"
        else [{"type": error.validator_value}]
    )
    descriptions = [_expected(option) for option in options]
    if None in descriptions:
        return _describe_other(error)
    expected = [e for d in descriptions if d is not None for e in d]
    message = f"expected {_join_or(expected)}, got {error.instance!r}."
    return [RecipeIssue(tuple(error.absolute_path), message)]


def _describe_too_few_items(
    error: jsonschema.ValidationError,
) -> list[RecipeIssue]:
    number = error.validator_value
    items = "item" if number == 1 else "items"
    message = f"expected at least {number} {items}, got {error.instance!r}."
    return [RecipeIssue(tuple(error.absolute_path), message)]


def _describe_out_of_range(
    error: jsonschema.ValidationError,
) -> list[RecipeIssue]:
    bound = "at least" if error.validator == "minimum" else "at most"
    message = (
        f"expected {bound} {error.validator_value}, got {error.instance!r}."
    )
    return [RecipeIssue(tuple(error.absolute_path), message)]


def _describe_other(error: jsonschema.ValidationError) -> list[RecipeIssue]:
    return [RecipeIssue(tuple(error.absolute_path), error.message)]


def _describe(
    error: jsonschema.ValidationError,
    alternatives: list[str],
) -> list[RecipeIssue]:
    """Describe a validation error in terms of the recipe."""
    if error.validator == "enum":
        return _describe_invalid_value(error, alternatives)
    describers = {
        "additionalProperties": _describe_unknown_keys,
        "anyOf": _describe_wrong_type,
        "maximum": _describe_out_of_range,
        "minItems": _describe_too_few_items,
        "minimum": _describe_out_of_range,
        "required": _describe_missing_keys,
        "type": _describe_wrong_type,
    }
    return describers.get(error.validator, _describe_other)(error)


def _mapping_items(
    node: yaml.MappingNode,
) -> list[tuple[yaml.Node, yaml.Node]]:
    """Get the items of a mapping, including those from merge keys (``<<``).

    The items are in the order that PyYAML uses to construct the mapping, so
    if a key occurs more than once, the last item is the one that is used.
    """
    merged = []
    items = []
    for key_node, value_node in node.value:
        if key_node.tag == "tag:yaml.org,2002:merge":
            sources = (
                value_node.value
                if isinstance(value_node, yaml.SequenceNode)
                else [value_node]
            )
            # In a sequence of mappings to merge, the first one has priority.
            for source in reversed(sources):
                if isinstance(source, yaml.MappingNode):
                    merged.extend(_mapping_items(source))
        else:
            items.append((key_node, value_node))
    return merged + items


def _find_child(
    node: yaml.Node,
    key: str | int,
) -> tuple[yaml.Node, yaml.Node] | None:
    """Find the key and value nodes of an item in a mapping or sequence.

    Merge keys (``<<``) are followed, so an item that comes from an anchor is
    found at the anchor.
    """
    found = None
    if isinstance(node, yaml.MappingNode):
        for key_node, value_node in _mapping_items(node):
            if isinstance(key_node, yaml.ScalarNode) and (
                key_node.value == str(key)
            ):
                found = key_node, value_node
    elif (
        isinstance(node, yaml.SequenceNode)
        and isinstance(key, int)
        and key < len(node.value)
    ):
        found = node.value[key], node.value[key]
    return found


def _find_position(
    root: yaml.Node,
    path: tuple[str | int, ...],
) -> yaml.Mark | None:
    """Find the position of a path in a recipe.

    For an item of a mapping, this is the position of its key.
    """
    node = root
    position = root.start_mark
    for key in path:
        found = _find_child(node, key)
        if found is None:
            return None
        key_node, node = found
        position = key_node.start_mark
    return position


def add_positions(issues: list[RecipeIssue], text: str) -> list[RecipeIssue]:
    """Add line and column numbers to issues and sort them by position."""
    try:
        root = yaml.compose(text, Loader=yaml.SafeLoader)
    except yaml.YAMLError:
        return issues
    if root is None:
        return issues
    located = []
    for issue in issues:
        position = _find_position(root, issue.path)
        located.append(
            issue
            if position is None
            else dataclasses.replace(
                issue,
                line=position.line + 1,
                column=position.column + 1,
            ),
        )
    return sorted(
        located,
        key=lambda i: (
            i.line is None,
            i.line or 0,
            i.column or 0,
            [str(key) for key in i.path],
        ),
    )


def validate_recipe(
    recipe: object,
    text: str | None = None,
) -> list[RecipeIssue]:
    """Validate a recipe against the JSON Schema.

    Parameters
    ----------
    recipe:
        The recipe, as loaded from YAML.
    text:
        The YAML text of the recipe. If given, the issues include line and
        column numbers and are sorted by their position in the text.

    Returns
    -------
    :
        The problems found. The list is empty if the recipe is valid.
    """
    issues: list[RecipeIssue] = []
    for error in _get_validator().iter_errors(_normalize(recipe)):
        for relevant, alternatives in _relevant_errors(error):
            for issue in _describe(relevant, alternatives):
                if issue not in issues:
                    issues.append(issue)
    if issues and text is not None:
        issues = add_positions(issues, text)
    return issues
