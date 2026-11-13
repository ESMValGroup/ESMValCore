"""JSON Schema for recipes.

The schema describes the structure of a recipe and the arguments of all
preprocessor functions. ESMValCore uses it to validate recipes, and editors can
use it to offer validation, completion, and documentation while writing a
recipe. The preprocessor part is generated from the signatures and docstrings
of the preprocessor functions.

Add a rule here if a JSON Schema can express it, so editors can report it too.
Checks of a recipe that the schema cannot express belong in
:mod:`esmvalcore._recipe.validate`, and checks that need the input data in
:mod:`esmvalcore._recipe.check`.
"""

from __future__ import annotations

import ast
import builtins
import collections.abc
import contextlib
import functools
import importlib
import inspect
import json
import re
import sys
import textwrap
import types
import typing
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import esmvalcore.preprocessor
from esmvalcore import __version__
from esmvalcore.config import CFG
from esmvalcore.config._diagnostics import DIAGNOSTICS, TAGS
from esmvalcore.preprocessor import (
    DEFAULT_ORDER,
    FINAL_STEPS,
    INITIAL_STEPS,
    MULTI_MODEL_FUNCTIONS,
)

if TYPE_CHECKING:
    from esmvalcore.config import Session

JSONSchema = dict[str, Any]

SCHEMA_DIALECT = "http://json-schema.org/draft-07/schema#"

# Schema keyword for the severity of a value that is not in an ``enum``, or
# of a key that is not in the ``properties`` of an object.
SEVERITY_KEYWORD = "x-severity"

# Preprocessor steps that can be used in a recipe. ESMValCore always applies
# the initial steps when it loads the data, ignoring the recipe.
RECIPE_STEPS = tuple(s for s in DEFAULT_ORDER if s not in INITIAL_STEPS)

# Preprocessor steps that support excluding datasets with ``exclude``.
EXCLUDE_STEPS = MULTI_MODEL_FUNCTIONS | {"weighting_landsea_fraction"}

# Arguments that ESMValCore fills in from the dataset if they are missing.
AUTOMATIC_ARGUMENTS = {
    "align_metadata": {"target_mip", "target_short_name"},
    "regrid_time": {"frequency"},
}

# Arguments that ESMValCore always sets, so they cannot be used in a recipe.
# For the final steps, this also includes all arguments without a default
# value.
INTERNAL_ARGUMENTS: dict[str, set[str]] = {
    step: {"session"} for step in FINAL_STEPS
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

_DATASET_NAME = (
    "the name of a dataset in the diagnostic, or `reference_dataset` or "
    "`alternative_dataset` for the dataset defined by that key in the variable"
)

# Arguments that ESMValCore converts when it reads the recipe. The type and
# description of the preprocessor function do not apply to these, because
# they describe the converted value.
RECIPE_ARGUMENTS: dict[str, dict[str, JSONSchema]] = {
    "extract_levels": {
        "levels": {
            "anyOf": [
                {"type": "number"},
                {"type": "array", "items": {"type": "number"}},
                {"type": "string"},
                {
                    "type": "object",
                    "properties": {"dataset": {"type": "string"}},
                    "required": ["dataset"],
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "properties": {
                        "cmor_table": {"type": "string"},
                        "coordinate": {"type": "string"},
                    },
                    "required": ["cmor_table", "coordinate"],
                    "additionalProperties": False,
                },
            ],
            "description": (
                "Target levels for the vertical interpolation. Either one or "
                "more levels, in the units of the vertical coordinate of the "
                f"data, the levels of a dataset given by {_DATASET_NAME} "
                "(also as `{dataset: <name>}`), or the levels of a coordinate "
                "in a CMOR table, e.g. `{cmor_table: CMIP6, coordinate: "
                "plev19}`. If the requested levels are sufficiently close to "
                "the levels of the data, the data is sliced instead of "
                "interpolated."
            ),
        },
    },
    "extract_shape": {
        "shapefile": {
            "type": "string",
            "description": (
                "Shapefile that defines the region(s) to extract. Either an "
                "absolute path, a path relative to the `auxiliary_data_dir`, "
                "or the name of a shapefile provided by ESMValCore: `ar6` for "
                "the IPCC WG1 reference regions (v4) used in Assessment "
                "Report 6 (https://doi.org/10.5281/zenodo.5176260). Use "
                "`ar6` with e.g. `ids: {Acronym: [GIC, WNA]}`."
            ),
        },
    },
    "regrid": {
        "target_grid": {
            "anyOf": [
                {"type": "string"},
                {
                    "type": "object",
                    "properties": {
                        key: {"type": "number"}
                        for key in (
                            "start_longitude",
                            "end_longitude",
                            "step_longitude",
                            "start_latitude",
                            "end_latitude",
                            "step_latitude",
                        )
                    },
                    "required": [
                        "start_longitude",
                        "end_longitude",
                        "step_longitude",
                        "start_latitude",
                        "end_latitude",
                        "step_latitude",
                    ],
                    "additionalProperties": False,
                },
            ],
            "description": (
                "Target grid. Either the grid of a dataset given by "
                f"{_DATASET_NAME}, a CORDEX domain name, e.g. `EUR-11`, a "
                "global regular grid given as `MxN`, the size of the grid "
                "cells in degrees longitude by latitude, e.g. `2.5x2.5`, or a "
                "regional regular grid given by `start_longitude`, "
                "`end_longitude`, `step_longitude`, `start_latitude`, "
                "`end_latitude`, and `step_latitude`."
            ),
        },
    },
}

# Operators of the statistics preprocessors that can be used in a recipe, see
# the table ``supported_stat_operator`` in ``doc/recipe/preprocessor.rst``. The
# other aggregators in :mod:`iris.analysis` require a Python function.
STATISTICS_OPERATORS = (
    "gmean",
    "hmean",
    "max",
    "mean",
    "median",
    "min",
    "peak",
    "percentile",
    "rms",
    "std_dev",
    "sum",
    "variance",
    "wpercentile",
)
_OPERATORS_TEXT = ", ".join(f"`{op}`" for op in STATISTICS_OPERATORS)
# Operators that require the keyword argument ``percent``.
PERCENT_OPERATORS = ("percentile", "wpercentile")
_PERCENT = {"type": "number", "minimum": 0, "maximum": 100}
PERCENT_SCHEMA: JSONSchema = {
    "anyOf": [_PERCENT, {"type": "array", "items": _PERCENT, "minItems": 1}],
    "description": (
        "Percentile or list of percentiles to compute, between 0 and 100. "
        "Required for the operators `percentile` and `wpercentile`."
    ),
}
_OPERATOR_TABLE = re.compile(
    r"given in\s+:ref:`this table\s+<supported_stat_operator>`",
)

# Labels that the docstrings of preprocessor functions refer to with :ref:,
# with the document that defines them and the title of their section, which
# is the text of a reference that does not specify one.
DOCUMENTATION_LABELS: dict[str, tuple[str, str | None]] = {
    "caching_regridding_weights": (
        "recipe/preprocessor",
        "Reusing regridding weights",
    ),
    "config-extra-facets": ("quickstart/configure", "Extra Facets"),
    "default regridding schemes": (
        "recipe/preprocessor",
        "Default regridding schemes",
    ),
    "generic regridding schemes": (
        "recipe/preprocessor",
        "Generic regridding schemes",
    ),
    "horizontal regridding": ("recipe/preprocessor", "Horizontal regridding"),
    "list_of_distance_metrics": ("recipe/preprocessor", None),
    "stat_preprocs": ("recipe/preprocessor", "Statistical preprocessors"),
    "supplementary_variables": (
        "recipe/overview",
        (
            "Defining supplementary variables (ancillary variables and cell "
            "measures)"
        ),
    ),
    "supported_stat_operator": ("recipe/preprocessor", None),
}

_REF_ROLE = re.compile(r":ref:`(?:([^`<]+?)\s+<([^`>]+)>|([^`]+))`")
_CODE_BLOCK = re.compile(
    r"^\.\. code-block::[ \t]*(\w*)\n[ \t]*\n((?:[ \t]+\S.*\n?|[ \t]*\n)+)",
    re.MULTILINE,
)

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


def _replace_ref(match: re.Match[str], *, markdown: bool) -> str:
    """Replace a :ref: role by its text, or by a link in Markdown."""
    label = (match[2] or match[3]).lower()
    document, title = DOCUMENTATION_LABELS.get(label, (None, None))
    text = match[1] or title or label
    if markdown and document is not None:
        anchor = re.sub(r"[^a-z0-9]+", "-", label)
        return f"[{text}]({_documentation_url(f'{document}.html#{anchor}')})"
    return text


def _clean_rst(text: str, *, markdown: bool = False) -> str:
    """Remove reStructuredText markup that does not read well in an editor.

    With ``markdown=True``, references to the documentation become links.
    """
    text = _REF_ROLE.sub(
        functools.partial(_replace_ref, markdown=markdown),
        text,
    )
    # Roles with an explicit target, e.g. :func:`text <target>`, are replaced
    # by their text, other roles, e.g. :math:`< 1`, by their content.
    text = re.sub(r":[\w:]+:`~?([^`<]+?)\s+<[^`>]+>`", r"\1", text)
    text = re.sub(r":[\w:]+:`~?([^`]*)`", r"\1", text)
    return text.replace("``", "`")


def _join_lines(text: str, *, markdown: bool) -> str:
    """Join the lines of each paragraph and clean up the markup."""
    paragraphs = re.split(r"\n[ \t]*\n", text.strip())
    return "\n\n".join(
        _clean_rst(
            " ".join(line.strip() for line in p.splitlines()),
            markdown=markdown,
        )
        for p in paragraphs
        if p.strip()
    )


def _rst_to_text(text: str, *, markdown: bool = False) -> str:
    """Convert a description in reStructuredText for display in an editor.

    Code blocks become fenced code blocks in Markdown and indented code
    otherwise.
    """
    parts = []
    position = 0
    for match in _CODE_BLOCK.finditer(text):
        parts.append(
            _join_lines(text[position : match.start()], markdown=markdown),
        )
        code = textwrap.dedent(match[2]).strip("\n")
        if markdown:
            parts.append(f"```{match[1]}\n{code}\n```")
        else:
            parts.append(textwrap.indent(code, "    "))
        position = match.end()
    parts.append(_join_lines(text[position:], markdown=markdown))
    return "\n\n".join(part for part in parts if part)


def _parse_docstring(docstring: str) -> tuple[str, dict[str, str]]:
    """Get the summary and parameter descriptions from a numpy docstring.

    These are returned in reStructuredText, use :func:`_rst_to_text` to
    display them in an editor.
    """
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
            parameters[name].append(line.rstrip())

    descriptions = {
        name: _OPERATOR_TABLE.sub(
            _OPERATORS_TEXT,
            textwrap.dedent("\n".join(text)).strip(),
        )
        for name, text in parameters.items()
    }
    return " ".join(summary), descriptions


def _parameter_schema(
    step: str,
    param: inspect.Parameter,
    description: str | None,
) -> JSONSchema:
    """Create a JSON Schema for an argument of a preprocessor function."""
    if param.name in RECIPE_ARGUMENTS.get(step, {}):
        return dict(RECIPE_ARGUMENTS[step][param.name])
    function = getattr(esmvalcore.preprocessor, step)
    annotation = _resolve_annotation(param.annotation, function)
    schema = _annotation_to_schema(annotation) or {}
    if description and _OPERATORS_TEXT in description:
        _add_operators(schema)
    if description:
        schema["description"] = _rst_to_text(description)
        markdown = _rst_to_text(description, markdown=True)
        if markdown != schema["description"]:
            schema["markdownDescription"] = markdown
    if param.default is not inspect.Parameter.empty:
        with contextlib.suppress(TypeError):
            schema["default"] = _to_json_value(param.default)
    return schema


def _add_operators(schema: JSONSchema) -> None:
    """Add the statistics operators to the schema of an argument."""
    for option in schema.get("anyOf", []):
        _add_operators(option)
    if schema.get("type") == "array":
        _add_operators(schema.get("items", {}))
    elif schema.get("type") == "string":
        schema["enum"] = list(STATISTICS_OPERATORS)
        schema["errorMessage"] = f"Supported operators: {_OPERATORS_TEXT}."
    elif schema.get("type") == "object":
        schema["properties"] = {"operator": {"type": "string"}}
        schema["required"] = ["operator"]
        _add_operators(schema["properties"]["operator"])
        _add_percent(schema)


def _add_percent(schema: JSONSchema) -> None:
    """Add the argument ``percent`` of the percentile operators."""
    schema["properties"]["percent"] = dict(PERCENT_SCHEMA)
    schema.setdefault("allOf", []).append(
        {
            "if": {
                "properties": {"operator": {"enum": list(PERCENT_OPERATORS)}},
                "required": ["operator"],
            },
            "then": {"required": ["percent"]},
        },
    )


def _add_operator_kwargs(settings: JSONSchema) -> None:
    """Add the keyword arguments of the operators to the step settings."""
    operator = settings["properties"].get("operator", {})
    if settings["additionalProperties"] and "enum" in operator:
        _add_percent(settings)


def _preprocessor_step_schema(step: str) -> JSONSchema:
    """Create a JSON Schema for the settings of a preprocessor step."""
    function = getattr(esmvalcore.preprocessor, step)
    summary, descriptions = _parse_docstring(inspect.getdoc(function) or "")
    parameters = list(inspect.signature(function).parameters.values())[1:]

    final_step = step in FINAL_STEPS
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
        if param.name in internal or (final_step and no_default):
            continue
        if no_default and param.name not in automatic:
            required.append(param.name)
        properties[param.name] = _parameter_schema(
            step,
            param,
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
    _add_operator_kwargs(settings)

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
    # Editors show the Markdown description instead of the description, with
    # a link to the documentation of using the step in a recipe.
    url = _recipe_documentation_url(step)
    return {
        "description": _rst_to_text(summary),
        "markdownDescription": (
            f"{_rst_to_text(summary, markdown=True)}\n\n"
            f"Refer to [how to use `{step}` in a recipe]({url})."
        ),
        "anyOf": options,
    }


def _documentation_url(path: str) -> str:
    """Get the URL of a page of the documentation."""
    # Only releases have their own documentation.
    version = (
        f"v{__version__}"
        if re.fullmatch(r"\d+\.\d+\.\d+", __version__)
        else "latest"
    )
    return (
        f"https://docs.esmvaltool.org/projects/ESMValCore/en/{version}/{path}"
    )


def _recipe_documentation_url(step: str) -> str:
    """Get the URL of the recipe documentation of a preprocessor step."""
    anchor = step.replace("_", "-")
    return _documentation_url(f"recipe/preprocessor.html#{anchor}")


# Places in the recipe documentation that several items link to.
_CORDEX_DOCS = ("CORDEX datasets", "cordex-datasets")
_DERIVATION_DOCS = ("how to derive a variable", "derive", "preprocessor")
_DIAGNOSTICS_DOCS = ("the `diagnostics` section of a recipe", "diagnostics")
_DOCUMENTATION_DOCS = (
    "the `documentation` section of a recipe",
    "recipe-documentation",
)
_PREPROCESSORS_DOCS = (
    "the `preprocessors` section of a recipe",
    "preprocessors",
)
_SCRIPTS_DOCS = ("how to run a diagnostic", "running-a-simple-diagnostic")


def _linked(
    schema: JSONSchema,
    topic: str,
    anchor: str,
    page: str = "overview",
) -> JSONSchema:
    """Add a link to the recipe documentation to the description of a schema.

    Editors show the ``markdownDescription`` instead of the ``description``.
    """
    url = _documentation_url(f"recipe/{page}.html#{anchor}")
    markdown = schema.get("markdownDescription", schema["description"])
    return {
        **schema,
        "markdownDescription": f"{markdown}\n\nRefer to [{topic}]({url}).",
    }


def _additional_datasets_schema(description: str) -> JSONSchema:
    """Create a JSON Schema for the ``additional_datasets`` of a recipe item."""
    return _linked(
        {
            "type": "array",
            "items": {"$ref": "#/definitions/dataset"},
            "description": description,
        },
        "how to use `additional_datasets` in a recipe",
        "diagnostic-and-variable-specific-datasets",
    )


def _preprocessor_schema() -> JSONSchema:
    """Create a JSON Schema for a preprocessor."""
    properties: JSONSchema = {
        "custom_order": _linked(
            {
                "type": "boolean",
                "default": False,
                "description": (
                    "Apply the preprocessor steps in the order in which they "
                    "are listed instead of the default order."
                ),
            },
            *_PREPROCESSORS_DOCS,
        ),
    }
    for step in RECIPE_STEPS:
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


class _Tags(typing.NamedTuple):
    """Tags that are defined by the installed ESMValTool."""

    tags: tuple[str, ...]
    hint: str
    """Explains what an unknown tag means and where to define it."""


def _known_tags() -> dict[str, _Tags]:
    """Get the tags that are defined by the installed ESMValTool.

    These are the tags in ``config-references.yml``, by section, and the
    references that have a bibtex file. Without ESMValTool, there are none.
    """
    tags = {
        section: _Tags(
            tuple(TAGS[section] or ()),
            f"Not defined in section '{section}' of "
            f"`{DIAGNOSTICS.tags_config}`, so the provenance will be "
            "incomplete.",
        )
        for section in ("authors", "projects", "realms", "themes")
        if section in TAGS
    }
    if DIAGNOSTICS.references.is_dir():
        tags["references"] = _Tags(
            tuple(
                sorted(
                    p.stem for p in DIAGNOSTICS.references.glob("*.bibtex")
                ),
            ),
            f"No bibtex file in `{DIAGNOSTICS.references}`, so the citation "
            "will be incomplete.",
        )
    return tags


def _tags_list(
    description: str,
    section: str,
    tags: dict[str, _Tags],
    min_items: int = 0,
) -> JSONSchema:
    """Create a JSON Schema for a list of tags or references."""
    schema = _string_list(description, min_items)
    if section in tags:
        # An unknown tag only makes the provenance less complete, so it is
        # reported as a warning. Editors also show it as a warning.
        schema["items"] = {
            "type": "string",
            "enum": list(tags[section].tags),
            "errorMessage": tags[section].hint,
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


def _markdown_description(markdown: str) -> JSONSchema:
    """Describe a schema, without markup for editors that do not show it."""
    description = markdown.replace("`", "")
    if description == markdown:
        return {"description": description}
    return {"description": description, "markdownDescription": markdown}


def _facet_value(description: str) -> JSONSchema:
    """Create a JSON Schema for :obj:`esmvalcore.typing.FacetValue`."""
    return {
        "anyOf": [
            {"type": ["string", "number"]},
            {"type": "array", "items": {"type": "string"}},
        ],
        **_markdown_description(description),
    }


# Observational datasets are described in the ESMValTool documentation.
_OBS_URL = "https://docs.esmvaltool.org/en/latest/input.html"
_OBS_TYPES = ("sat", "reanaly", "ground", "clim", "campaign")
_TYPES = {"OBS": _OBS_TYPES, "OBS6": _OBS_TYPES}


def _tier_schema() -> JSONSchema:
    """Create a JSON Schema for the tier of an observational dataset."""
    description = "Tier of an observational dataset"
    return {
        "type": "integer",
        "minimum": 1,
        "maximum": 3,
        "description": f"{description}.",
        "markdownDescription": (
            f"{description}: 2 for freely available datasets and 3 for "
            "restricted datasets. Refer to the "
            f"[tiers of observational datasets]({_OBS_URL}#tiers)."
        ),
    }


def _type_schema() -> JSONSchema:
    """Create a JSON Schema for the type of an observational dataset."""
    schema = _facet_value(
        "Type of an observational dataset: "
        f"{_join_or(list(_OBS_TYPES))} for the OBS and OBS6 projects.",
    )
    types = ", ".join(f"`{t}`" for t in _OBS_TYPES)
    schema["markdownDescription"] = (
        f"Type of an observational dataset: {types} for the OBS and OBS6 "
        "projects. Refer to the [types of observational datasets]"
        f"({_OBS_URL}#observation-types)."
    )
    return schema


def _facets_schema(projects: list[str]) -> JSONSchema:
    """Create a JSON Schema for the facets of a dataset or variable.

    See doc/reference/facets.rst for the facets that ESMValCore uses.
    """
    year = {"type": "integer", "minimum": 0, "maximum": 10000}
    project: JSONSchema = {
        "type": "string",
        **_markdown_description("Project, for example `CMIP6`."),
    }
    if projects:
        # Without a project in the configuration, ESMValCore cannot find the
        # data.
        project["enum"] = projects
        project["errorMessage"] = (
            "Not defined in the `projects` section of the configuration."
        )
    return {
        "type": "object",
        "properties": {
            "dataset": {
                "type": "string",
                "description": "Name of the dataset.",
            },
            "project": project,
            "short_name": {
                "type": "string",
                "description": "Short name of the variable.",
            },
            "mip": {
                "type": "string",
                **_markdown_description(
                    "MIP table, for example `Amon` for CMIP6 or `atmos` for "
                    "CMIP7.",
                ),
            },
            "exp": _string_or_list(
                "Experiment, or a list of experiments to concatenate.",
            ),
            "ensemble": _linked(
                _string_or_list("Ensemble member(s)."),
                "the short notation of ensemble members",
                "short-notation-of-ensemble-members-and-sub-experiments",
            ),
            "activity": _facet_value(
                "Activity, for example `CMIP` or `ScenarioMIP`.",
            ),
            "branding_suffix": _facet_value(
                "Variable branding suffix (CMIP7).",
            ),
            "domain": _linked(
                _facet_value("CORDEX domain, for example `EUR-11`."),
                *_CORDEX_DOCS,
            ),
            "driver": _linked(
                _facet_value("Driving global model of a CORDEX dataset."),
                *_CORDEX_DOCS,
            ),
            "frequency": _facet_value("Frequency, for example `mon`."),
            "grid": _facet_value("Grid label, for example `gn`."),
            "institute": _facet_value("Institute that produced the data."),
            "product": _facet_value("Product (CMIP5)."),
            "rcm_version": _linked(
                _facet_value(
                    "Version of the regional climate model (CORDEX).",
                ),
                *_CORDEX_DOCS,
            ),
            "realm": _facet_value("Realm, for example `atmos`."),
            "region": _facet_value("Region (CMIP7)."),
            "version": _facet_value("Version of the dataset."),
            "timerange": _linked(
                {
                    "type": ["string", "integer"],
                    **_markdown_description(
                        "Time range in ISO 8601 format, for example "
                        "`1990/2000`.",
                    ),
                },
                "examples of time ranges",
                "timerange-examples",
            ),
            "start_year": {
                **year,
                "description": "First year. Alternative to `timerange`.",
            },
            "end_year": {
                **year,
                "description": "Last year. Alternative to `timerange`.",
            },
            "alias": _linked(
                {
                    "type": "string",
                    "description": (
                        "Name of the dataset to use in diagnostics."
                    ),
                },
                "how to use `alias` in a recipe",
                "dataset-alias",
            ),
            "tier": _tier_schema(),
            "type": _type_schema(),
            "derive": _linked(
                {
                    "type": "boolean",
                    "description": (
                        "Derive the variable from other variables."
                    ),
                },
                *_DERIVATION_DOCS,
            ),
            "force_derivation": _linked(
                {
                    "type": "boolean",
                    "description": (
                        "Derive the variable even if it is available directly."
                    ),
                },
                *_DERIVATION_DOCS,
            ),
            "supplementary_variables": _linked(
                {
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
                "how to define supplementary variables",
                "supplementary-variables",
            ),
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
            "preprocessor": _linked(
                {
                    "type": "string",
                    "description": "Name of the preprocessor to apply.",
                },
                *_PREPROCESSORS_DOCS,
            ),
            "reference_dataset": {
                "type": "string",
                "description": "Reference dataset for this variable.",
            },
            "alternative_dataset": {
                "type": "string",
                "description": "Alternative dataset for this variable.",
            },
            "additional_datasets": _additional_datasets_schema(
                "Datasets to use for this variable only.",
            ),
        },
    }


def _script_schema() -> JSONSchema:
    return {
        "type": "object",
        "properties": {
            "script": _linked(
                {
                    "type": "string",
                    "description": (
                        "Path to the diagnostic script, relative to the "
                        "diagnostics directory or absolute."
                    ),
                },
                *_SCRIPTS_DOCS,
            ),
            "ancestors": _linked(
                _string_list(
                    "Tasks whose output this script uses, as "
                    "`diagnostic/variable_group` or `diagnostic/script`. "
                    "Wildcards are allowed.",
                ),
                "ancestor tasks",
                "ancestor-tasks",
            ),
        },
        "required": ["script"],
    }


def _diagnostic_schema(tags: dict[str, _Tags]) -> JSONSchema:
    return {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Title."},
            "description": {
                "type": ["string", "null"],
                "description": "Description of the diagnostic.",
            },
            "themes": _linked(
                _tags_list("Themes.", "themes", tags),
                *_DIAGNOSTICS_DOCS,
            ),
            "realms": _linked(
                _tags_list("Realms.", "realms", tags),
                *_DIAGNOSTICS_DOCS,
            ),
            "variables": _linked(
                {
                    "type": ["object", "null"],
                    "additionalProperties": {
                        "anyOf": [
                            {"$ref": "#/definitions/variable"},
                            {"type": "null"},
                        ],
                    },
                    "description": "Variable groups to preprocess.",
                },
                "how variables and datasets are combined",
                "variable-and-dataset-definitions",
            ),
            "scripts": _linked(
                {
                    "type": ["object", "null"],
                    "additionalProperties": {"$ref": "#/definitions/script"},
                    "description": "Diagnostic scripts to run.",
                },
                *_SCRIPTS_DOCS,
            ),
            "additional_datasets": _additional_datasets_schema(
                "Datasets to use for this diagnostic only.",
            ),
        },
        "required": ["scripts"],
        "additionalProperties": False,
        # Unknown keys are deprecated, so they are only a warning for now.
        SEVERITY_KEYWORD: "warning",
    }


def _documentation_schema(tags: dict[str, _Tags]) -> JSONSchema:
    return _linked(
        {
            "type": "object",
            # This is here because a draft-07 $ref ignores the keys next to it.
            "description": (
                "Title, description, authors, and other details of the recipe."
            ),
            "properties": {
                "title": {"type": "string", "description": "Recipe title."},
                "description": {
                    "type": "string",
                    "description": "Description of the recipe.",
                },
                "authors": _linked(
                    _tags_list(
                        "Authors, as defined in `config-references.yml`.",
                        "authors",
                        tags,
                        min_items=1,
                    ),
                    *_DOCUMENTATION_DOCS,
                ),
                "maintainer": _linked(
                    _tags_list(
                        "Maintainers, as defined in `config-references.yml`.",
                        "authors",
                        tags,
                    ),
                    *_DOCUMENTATION_DOCS,
                ),
                "projects": _linked(
                    _tags_list(
                        "Research projects that funded the development of "
                        "the recipe, as defined in `config-references.yml`.",
                        "projects",
                        tags,
                    ),
                    *_DOCUMENTATION_DOCS,
                ),
                "references": _linked(
                    _tags_list(
                        "References, as bibtex keys.",
                        "references",
                        tags,
                    ),
                    *_DOCUMENTATION_DOCS,
                ),
            },
            "required": ["title", "description", "authors"],
            "additionalProperties": False,
            # Unknown keys are deprecated, so they are only a warning for now.
            SEVERITY_KEYWORD: "warning",
        },
        *_DOCUMENTATION_DOCS,
    )


def generate_recipe_schema(session: Session | None = None) -> JSONSchema:
    """Generate a JSON Schema for recipes.

    The schema only allows the projects in the configuration. If ESMValTool is
    installed, it also lists the provenance tags and references that it
    defines, refer to :ref:`config-ref`.

    Parameters
    ----------
    session:
        The session that defines the projects. If not
        given, the configuration in :obj:`esmvalcore.config.CFG` is used.

    Returns
    -------
    :
        A JSON Schema (draft-07) that describes a recipe.
    """
    config = CFG if session is None else session
    return _recipe_schema(_known_tags(), list(config["projects"]))


def _recipe_schema(
    tags: dict[str, _Tags],
    projects: list[str],
) -> JSONSchema:
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
            "datasets": _linked(
                {
                    "type": "array",
                    "items": {"$ref": "#/definitions/dataset"},
                    "description": "Datasets to use in all diagnostics.",
                },
                "the `datasets` section of a recipe",
                "datasets",
            ),
            "preprocessors": _linked(
                {
                    "type": "object",
                    "additionalProperties": {
                        "$ref": "#/definitions/preprocessor",
                    },
                    "description": (
                        "Preprocessors that variables can refer to."
                    ),
                },
                *_PREPROCESSORS_DOCS,
            ),
            "diagnostics": _linked(
                {
                    "type": "object",
                    "additionalProperties": {
                        "$ref": "#/definitions/diagnostic",
                    },
                    "description": "Diagnostics to run.",
                },
                *_DIAGNOSTICS_DOCS,
            ),
        },
        "required": ["documentation", "diagnostics"],
        "definitions": {
            "documentation": _documentation_schema(tags),
            "facets": _facets_schema(projects),
            "dataset": _dataset_schema(),
            "variable": _variable_schema(),
            "script": _script_schema(),
            "diagnostic": _diagnostic_schema(tags),
            "preprocessor": _preprocessor_schema(),
            "steps": {
                step: _preprocessor_step_schema(step) for step in RECIPE_STEPS
            },
        },
    }


def _join_or(items: list[str]) -> str:
    """Join items like "a, b, or c"."""
    if len(items) <= 2:
        return " or ".join(items)
    return ", ".join(items[:-1]) + ", or " + items[-1]
