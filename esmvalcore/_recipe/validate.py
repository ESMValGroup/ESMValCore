"""Validate recipes before the data is searched.

:func:`validate_recipe` checks a recipe against the JSON Schema from
:mod:`esmvalcore._recipe.json_schema` and for problems that the schema cannot
express, such as preprocessors that are used but not defined, or keys that are
defined more than once. These checks only need the recipe and the
configuration, from the session if one is given, including the CMOR tables.
That makes them fast, and all problems are reported at once, with their
position in the recipe file.

Add a check here if it finds out whether the recipe itself is correct, unless
the schema can express it. Checks of whether the recipe can run on this
machine, for example because they need the input data, other files on disk
such as shapefiles, or the installed software, belong in
:mod:`esmvalcore._recipe.check`.
"""

from __future__ import annotations

import collections.abc
import dataclasses
import difflib
import fnmatch
import inspect
import logging
import re
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import jsonschema
import yaml

import esmvalcore.preprocessor
from esmvalcore._recipe import check
from esmvalcore._recipe._io import _parse_yaml
from esmvalcore._recipe.json_schema import (
    _TYPES,
    SEVERITY_KEYWORD,
    STATISTICS_OPERATORS,
    JSONSchema,
    _documentation_url,
    _join_or,
    generate_recipe_schema,
)
from esmvalcore.config import CFG
from esmvalcore.config._tasksep import TASKSEP
from esmvalcore.exceptions import ESMValCoreDeprecationWarning, RecipeError
from esmvalcore.preprocessor import INITIAL_STEPS
from esmvalcore.preprocessor._regrid import _load_generic_scheme
from esmvalcore.preprocessor._shared import get_iris_aggregator

if TYPE_CHECKING:
    from collections.abc import Iterable

    from esmvalcore.config import Config, Session

logger = logging.getLogger(__name__)


def _location(path: Iterable[str | int]) -> str:
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


def yaml_error_issue(exc: yaml.YAMLError) -> RecipeIssue:
    """Describe a YAML syntax error as an issue."""
    if isinstance(exc, yaml.MarkedYAMLError):
        mark = exc.problem_mark or exc.context_mark
        return RecipeIssue(
            (),
            f"invalid YAML: {exc.problem or exc.context}.",
            line=None if mark is None else mark.line + 1,
            column=None if mark is None else mark.column + 1,
        )
    return RecipeIssue((), f"invalid YAML: {exc}.")


TYPE_DESCRIPTIONS = {
    "array": "a list",
    "boolean": "true or false",
    "integer": "an integer",
    "null": "nothing",
    "number": "a number",
    "object": "a mapping",
    "string": "a string",
}


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
        if len(options) > 1:
            # Several options accept values of this type, e.g. mappings with
            # different keys. Use the option that the value matches best.
            counts = collections.Counter(
                e.relative_schema_path[0] for e in error.context
            )
            fewest = min(counts[i] for i in options)
            options = [i for i in options if counts[i] == fewest]
        if len(options) == 1:
            others = [
                description
                for i, option in enumerate(error.validator_value)
                if i != options[0]
                for description in _expected(option) or []
            ]
            others = list(dict.fromkeys(others))
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
    in_preprocessor = len(path) == 2 and path[0] == "preprocessors"
    if in_preprocessor:
        problem = "unknown preprocessor function."
    elif _in_step(path):
        problem = "unknown argument."
    else:
        problem = "unknown key."
    severity = error.schema.get(SEVERITY_KEYWORD, "error")
    issues = []
    for key in error.instance:
        if key in known:
            continue
        if in_preprocessor and key in INITIAL_STEPS:
            message = (
                "this preprocessor function cannot be used in a recipe, "
                "because it is always applied when the data is loaded."
            )
            issues.append(RecipeIssue((*path, key), message))
            continue
        message = problem + _suggest(key, known)
        if _in_step(path):
            valid = ", ".join(sorted(known)) or "none"
            message += f" Valid arguments are: {valid}."
        if severity == "warning":
            message += (
                " Unknown keys are ignored, this is deprecated and will not "
                "work in v2.18.0."
            )
        issues.append(RecipeIssue((*path, key), message, severity=severity))
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
    expected = list(dict.fromkeys(expected))
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
        if not _allows_type(error.schema, error.instance):
            # The value also has the wrong type, which is reported instead.
            return []
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


# Positions of aliases, by the id of the node that contains the alias and
# the index of the alias in a sequence or the id of its key in a mapping.
_Aliases = dict[tuple[int, int], yaml.Mark]


@dataclasses.dataclass
class _Composed:
    """A tree of nodes with the positions and anchor names of aliases."""

    root: yaml.Node
    aliases: _Aliases
    anchors: dict[tuple[int, int], str]


class _AliasLoader(yaml.SafeLoader):
    """Loader that records where aliases are used.

    An alias composes to the node of its anchor, so the node itself does not
    tell where the alias is.
    """

    def __init__(self, stream: str) -> None:
        super().__init__(stream)
        self.aliases: _Aliases = {}
        self.anchor_names: dict[tuple[int, int], str] = {}

    def compose_node(
        self,
        parent: yaml.Node | None,
        index: int,
    ) -> yaml.Node | None:
        # The index is the position in a sequence or, despite the type stubs
        # of PyYAML, the key node in a mapping, or None for a key or the root
        # node.
        if (
            parent is not None
            and index is not None
            and self.check_event(yaml.AliasEvent)
        ):
            child = index if isinstance(index, int) else id(index)
            event = self.peek_event()
            self.aliases[id(parent), child] = event.start_mark
            self.anchor_names[id(parent), child] = event.anchor
        return super().compose_node(parent, index)


def _mapping_items(
    node: yaml.MappingNode,
    aliases: _Aliases,
) -> list[tuple[yaml.Node, yaml.Node, yaml.Mark | None]]:
    """Get the items of a mapping, including those from merge keys (``<<``).

    The items are in the order that PyYAML uses to construct the mapping, so
    if a key occurs more than once, the last item is the one that is used.
    Each item comes with the position of the alias that merged it into the
    mapping, if any.
    """
    merged: list[tuple[yaml.Node, yaml.Node, yaml.Mark | None]] = []
    items = []
    for key_node, value_node in node.value:
        if key_node.tag == "tag:yaml.org,2002:merge":
            sources = (
                [
                    (source, aliases.get((id(value_node), i)))
                    for i, source in enumerate(value_node.value)
                ]
                if isinstance(value_node, yaml.SequenceNode)
                else [(value_node, aliases.get((id(node), id(key_node))))]
            )
            # In a sequence of mappings to merge, the first one has priority.
            for source, alias in reversed(sources):
                if isinstance(source, yaml.MappingNode):
                    merged.extend(
                        (key, value, alias or inner)
                        for key, value, inner in _mapping_items(
                            source,
                            aliases,
                        )
                    )
        else:
            items.append((key_node, value_node, None))
    return merged + items


def _find_child(
    node: yaml.Node,
    key: str | int,
    aliases: _Aliases,
) -> tuple[yaml.Mark, yaml.Node, yaml.Mark | None] | None:
    """Find an item in a mapping or sequence.

    Returns the position of the item, its value node, and the position of
    the alias through which the value is reached, if any. Merge keys (``<<``)
    are followed, so an item that comes from an anchor is found at the alias
    that merges it.
    """
    found = None
    if isinstance(node, yaml.MappingNode):
        for key_node, value_node, merged in _mapping_items(node, aliases):
            if isinstance(key_node, yaml.ScalarNode) and (
                key_node.value == str(key)
            ):
                alias = merged or aliases.get((id(node), id(key_node)))
                found = merged or key_node.start_mark, value_node, alias
    elif (
        isinstance(node, yaml.SequenceNode)
        and isinstance(key, int)
        and key < len(node.value)
    ):
        item = node.value[key]
        alias = aliases.get((id(node), key))
        found = alias or item.start_mark, item, alias
    return found


def _find_position(
    root: yaml.Node,
    aliases: _Aliases,
    path: tuple[str | int, ...],
) -> yaml.Mark | None:
    """Find the position of a path in a recipe.

    For an item of a mapping, this is the position of its key. For an item
    inside an anchored node, this is the position of the alias that leads to
    it, so the problem is shown where the anchored node is used.
    """
    node = root
    position = root.start_mark
    via_alias = None
    for key in path:
        found = _find_child(node, key, aliases)
        if found is None:
            return None
        position, node, alias = found
        position = via_alias or position
        via_alias = via_alias or alias
    return position


def _compose(text: str) -> _Composed | None:
    """Parse YAML text into a tree of nodes and the positions of aliases.

    Returns None if that fails or the text is empty.
    """
    loader = _AliasLoader(text)
    try:
        root = loader.get_single_node()
    except yaml.YAMLError:
        return None
    finally:
        loader.dispose()
    if root is None:
        return None
    return _Composed(root, loader.aliases, loader.anchor_names)


_MERGE_TAG = "tag:yaml.org,2002:merge"


@dataclasses.dataclass
class _YAMLChecker:
    """Find YAML that is ignored, deprecated, or ambiguous in a recipe."""

    composed: _Composed
    issues: list[RecipeIssue] = dataclasses.field(default_factory=list)
    visited: set[int] = dataclasses.field(default_factory=set)
    ancestors: list[int] = dataclasses.field(default_factory=list)

    def check(self, node: yaml.Node, path: tuple[str | int, ...]) -> None:
        """Check a node and the nodes it contains."""
        if id(node) in self.visited:
            # Report problems in an anchored node only once.
            return
        self.visited.add(id(node))
        self.ancestors.append(id(node))
        if isinstance(node, yaml.MappingNode):
            self._check_mapping(node, path)
        elif isinstance(node, yaml.SequenceNode):
            for i, item in enumerate(node.value):
                self._check_child(node, i, item, (*path, i))
        elif isinstance(node, yaml.ScalarNode):
            self._check_scalar(node, path)
        self.ancestors.pop()

    def _check_child(
        self,
        parent: yaml.Node,
        child: int,
        node: yaml.Node,
        path: tuple[str | int, ...],
    ) -> None:
        """Check a child node, unless it is an alias to one of its parents."""
        if id(node) not in self.ancestors:
            self.check(node, path)
            return
        mark = self.composed.aliases[id(parent), child]
        name = self.composed.anchors[id(parent), child]
        self.issues.append(
            _yaml_issue(
                path,
                f"alias '*{name}' refers to the anchor '&{name}' that "
                "contains it, which other YAML parsers do not accept. This "
                "is deprecated and will not work in v2.18.0: remove the "
                "alias.",
                mark,
            ),
        )

    def _check_mapping(
        self,
        node: yaml.MappingNode,
        path: tuple[str | int, ...],
    ) -> None:
        """Check the keys and values of a mapping."""
        merges = [(k, v) for k, v in node.value if k.tag == _MERGE_TAG]
        if len(merges) > 1:
            self._report_merge_keys(node, merges, path)
        for key_node, value_node in merges:
            self._check_merged(node, key_node, value_node, path)
        items = [
            (key_node, value_node)
            for key_node, value_node in node.value
            if isinstance(key_node, yaml.ScalarNode)
            and key_node.tag != _MERGE_TAG
        ]
        last = {key_node.value: key_node for key_node, _ in items}
        for key_node, value_node in items:
            key = key_node.value
            self._check_scalar(key_node, (*path, key))
            used = last[key]
            if used is key_node:
                self._check_child(node, id(key_node), value_node, (*path, key))
            else:
                # The ignored value is not checked any further.
                self.issues.append(
                    _yaml_issue(
                        (*path, key),
                        f"duplicate key, this definition is ignored because "
                        f"'{key}' is defined again on line "
                        f"{used.start_mark.line + 1}.",
                        key_node.start_mark,
                    ),
                )

    def _check_merged(
        self,
        node: yaml.MappingNode,
        key_node: yaml.Node,
        value_node: yaml.Node,
        path: tuple[str | int, ...],
    ) -> None:
        """Check the mappings that a merge key merges."""
        if isinstance(value_node, yaml.SequenceNode):
            if id(value_node) not in self.visited:
                self.visited.add(id(value_node))
                for i, item in enumerate(value_node.value):
                    self._check_child(value_node, i, item, path)
        else:
            self._check_child(node, id(key_node), value_node, path)

    def _report_merge_keys(
        self,
        node: yaml.MappingNode,
        merges: list[tuple[yaml.Node, yaml.Node]],
        path: tuple[str | int, ...],
    ) -> None:
        """Report merge keys that occur more than once in a mapping."""
        # The mappings of later merge keys have priority, while the first
        # mapping in a sequence of mappings to merge has priority.
        sources: list[str] = []
        for key_node, value_node in reversed(merges):
            if isinstance(value_node, yaml.SequenceNode):
                sources.extend(
                    self._source(value_node, i)
                    for i in range(len(value_node.value))
                )
            else:
                sources.append(self._source(node, id(key_node)))
        self.issues.append(
            _yaml_issue(
                path,
                "duplicate merge key '<<', which other YAML parsers do not "
                "accept. This is deprecated and will not work in v2.18.0: use "
                "one merge key with the mappings in this order: "
                f"<<: [{', '.join(sources)}]",
                merges[1][0].start_mark,
            ),
        )

    def _source(self, parent: yaml.Node, child: int) -> str:
        """Describe a mapping to merge by its alias."""
        name = self.composed.anchors.get((id(parent), child))
        return "{...}" if name is None else f"*{name}"

    def _check_scalar(
        self,
        node: yaml.ScalarNode,
        path: tuple[str | int, ...],
    ) -> None:
        """Report values that YAML 1.1 and 1.2 parsers read differently."""
        if node.style is not None:
            return
        message = _ambiguous_scalar(node.value, node.tag)
        if message is not None:
            self.issues.append(_yaml_issue(path, message, node.start_mark))


def _yaml_issue(
    path: tuple[str | int, ...],
    message: str,
    mark: yaml.Mark,
) -> RecipeIssue:
    return RecipeIssue(
        path,
        message,
        line=mark.line + 1,
        column=mark.column + 1,
        severity="warning",
    )


# The types of plain scalars in the core schema of YAML 1.2, see
# https://yaml.org/spec/1.2.2/#1032-tag-resolution.
_YAML_12_TYPES = {
    "null": re.compile(r"null|Null|NULL|~|"),
    "a boolean": re.compile(r"true|True|TRUE|false|False|FALSE"),
    "an integer": re.compile(r"[-+]?[0-9]+|0o[0-7]+|0x[0-9a-fA-F]+"),
    "a number": re.compile(
        r"[-+]?(\.[0-9]+|[0-9]+(\.[0-9]*)?)([eE][-+]?[0-9]+)?"
        r"|[-+]?\.(inf|Inf|INF)|\.nan|\.NaN|\.NAN",
    ),
}

# The types of plain scalars in YAML 1.1, as read by PyYAML. Dates are not
# included, because YAML 1.2 has no other way to write them.
_YAML_11_TYPES = {
    "tag:yaml.org,2002:null": "null",
    "tag:yaml.org,2002:bool": "a boolean",
    "tag:yaml.org,2002:int": "an integer",
    "tag:yaml.org,2002:float": "a number",
    "tag:yaml.org,2002:str": "text",
}


def _yaml_12_integer(value: str) -> int:
    """Read an integer as YAML 1.2 does."""
    if value.startswith(("0o", "0x")):
        return int(value[2:], 8 if value[1] == "o" else 16)
    return int(value)


def _dump(value: object) -> str:
    """Write a value in a way that YAML 1.1 and 1.2 read the same."""
    return yaml.safe_dump(value).split("\n")[0]


def _ambiguous_scalar(value: str, tag: str) -> str | None:
    """Describe a plain scalar that YAML 1.1 and 1.2 read differently."""
    old = _YAML_11_TYPES.get(tag)
    new = next(
        (t for t, regex in _YAML_12_TYPES.items() if regex.fullmatch(value)),
        "text",
    )
    if old is None or (
        old == new
        and (
            old != "an integer"
            or yaml.safe_load(value) == _yaml_12_integer(value)
        )
    ):
        return None
    if old == "text":
        hint = "quote it if it is text, or write " + (
            f"{_dump(float(value))} for the number"
            if new == "a number"
            else f"{_yaml_12_integer(value)} for the integer"
        )
    else:
        hint = (
            f"write {_dump(yaml.safe_load(value))} for {old}, "
            "or quote it if it is text"
        )
    if old == new:
        # Octal integers are written differently in YAML 1.2.
        read = f"{yaml.safe_load(value)}", f"{_yaml_12_integer(value)}"
    else:
        read = old, new
    return (
        f"ambiguous value '{value}', read as {read[0]} by ESMValCore but as "
        f"{read[1]} by YAML 1.2 parsers, for example in editors. This is "
        f"deprecated and will not work in v2.18.0: {hint}."
    )


def _mapping(data: object, key: str) -> dict:
    """Get a mapping from a recipe, or an empty one if it is not there."""
    value = data.get(key) if isinstance(data, dict) else None
    return value if isinstance(value, dict) else {}


def _list(data: object, key: str) -> list:
    """Get a list from a recipe, or an empty one if it is not there."""
    value = data.get(key) if isinstance(data, dict) else None
    return value if isinstance(value, list) else []


def _preprocessor_facets(
    recipe: object,
) -> collections.abc.Iterator[tuple[tuple[str | int, ...], object]]:
    """Find the values of the ``preprocessor`` facet in a recipe.

    Datasets can override the preprocessor of their variable group.
    """
    entries: list[tuple[tuple[str | int, ...], object]] = [
        (("datasets", i), entry)
        for i, entry in enumerate(_list(recipe, "datasets"))
    ]
    for diagnostic_name, diagnostic in _mapping(recipe, "diagnostics").items():
        path: tuple[str | int, ...] = ("diagnostics", diagnostic_name)
        entries.extend(
            ((*path, "additional_datasets", i), entry)
            for i, entry in enumerate(_list(diagnostic, "additional_datasets"))
        )
        for group, variable in _mapping(diagnostic, "variables").items():
            group_path = (*path, "variables", group)
            # The schema reports preprocessors of variable groups that are
            # not a string.
            name = (
                variable.get("preprocessor")
                if isinstance(variable, dict)
                else None
            )
            if isinstance(name, str):
                yield (*group_path, "preprocessor"), name
            entries.extend(
                ((*group_path, "additional_datasets", i), entry)
                for i, entry in enumerate(
                    _list(variable, "additional_datasets"),
                )
            )
    for entry_path, entry in entries:
        if isinstance(entry, dict) and "preprocessor" in entry:
            yield (*entry_path, "preprocessor"), entry["preprocessor"]


def _matches(pattern: str, value: object) -> bool:
    """Check if a facet value matches a pattern of the extra facets.

    Values that are not given or not a string may match any pattern.
    """
    return not isinstance(value, str) or fnmatch.fnmatchcase(value, pattern)


def _extra_facet_names(
    project: str,
    facets: dict,
    config: Config | Session,
) -> set[str]:
    """Get the names of the extra facets that may apply to a dataset."""
    projects = config["projects"]
    extra_facets = projects.get(project, {}).get("extra_facets", {})
    return {
        name
        for dataset, mips in extra_facets.items()
        if _matches(dataset, facets.get("dataset"))
        for mip, variables in mips.items()
        if _matches(mip, facets.get("mip"))
        for short_name, values in variables.items()
        if _matches(short_name, facets.get("short_name"))
        for name in values
    }


def _variable_groups(
    recipe: object,
) -> collections.abc.Iterator[
    tuple[
        tuple[str | int, ...],
        dict,
        list[tuple[tuple[str | int, ...], dict]],
    ]
]:
    """Find the variable groups with the datasets that they use."""
    datasets = [
        (("datasets", i), entry)
        for i, entry in enumerate(_list(recipe, "datasets"))
    ]
    for diagnostic_name, diagnostic in _mapping(recipe, "diagnostics").items():
        path: tuple[str | int, ...] = ("diagnostics", diagnostic_name)
        diagnostic_datasets = datasets + [
            ((*path, "additional_datasets", i), entry)
            for i, entry in enumerate(_list(diagnostic, "additional_datasets"))
        ]
        for group, variable in _mapping(diagnostic, "variables").items():
            group_path = (*path, "variables", group)
            additional = _list(variable, "additional_datasets")
            group_datasets = diagnostic_datasets + [
                ((*group_path, "additional_datasets", i), entry)
                for i, entry in enumerate(additional)
            ]
            yield (
                group_path,
                variable if isinstance(variable, dict) else {},
                [(p, e) for p, e in group_datasets if isinstance(e, dict)],
            )


def _datasets(
    recipe: object,
) -> collections.abc.Iterator[
    tuple[tuple[str | int, ...], dict, dict[str, tuple[str | int, ...]]]
]:
    """Find the datasets of each variable group with their facets.

    The facets of a dataset override those of its variable group. Also
    yields the path of the dataset or variable group that defines each facet.
    """
    for group_path, facets, group_datasets in _variable_groups(recipe):
        for dataset_path, entry in group_datasets:
            origin = {
                **dict.fromkeys(facets, group_path),
                **dict.fromkeys(entry, dataset_path),
            }
            yield dataset_path, {**facets, **entry}, origin


def _find_incomplete_datasets(recipe: object) -> list[RecipeIssue]:
    """Find problems that prevent reading the datasets from a recipe.

    These are variable groups without datasets, datasets that occur more than
    once in a variable group, and datasets without a dataset, project, or MIP.
    """
    issues: list[RecipeIssue] = []

    def add(issue: RecipeIssue) -> None:
        # Datasets in the `datasets` section belong to every variable group.
        if issue not in issues:
            issues.append(issue)

    for group_path, variable, group_datasets in _variable_groups(recipe):
        if not group_datasets:
            add(
                RecipeIssue(
                    group_path,
                    "no datasets. Add them to the `datasets` section or to "
                    "`additional_datasets`.",
                ),
            )
        previous = []
        for path, entry in group_datasets:
            if entry in previous:
                add(RecipeIssue(path, "duplicate dataset."))
            previous.append(entry)
            facets = {**variable, **entry}
            for key in ("dataset", "project", "mip"):
                if key not in facets:
                    add(RecipeIssue(path, f"missing required key {key!r}."))
    return issues


def _find_unknown_reference_datasets(recipe: object) -> list[RecipeIssue]:
    """Find reference and alternative datasets that are not in the group.

    Datasets can be referred to by their name or alias. An empty value means
    that there is no such dataset, and some diagnostic scripts accept several
    datasets separated by ``|``.
    """
    issues = []
    for group_path, variable, group_datasets in _variable_groups(recipe):
        values = [entry.get("dataset") for _, entry in group_datasets]
        datasets = [v for v in values if isinstance(v, str)]
        if len(datasets) < len(values) or any(
            char in name for name in datasets for char in "*?["
        ):
            # Patterns and lists of datasets may match the dataset.
            continue
        aliases = [entry.get("alias") for _, entry in group_datasets]
        names = {*datasets, *(a for a in aliases if isinstance(a, str))}
        for key in ("reference_dataset", "alternative_dataset"):
            value = variable.get(key)
            if not isinstance(value, str) or not value:
                continue
            for name in value.split("|"):
                if name not in names:
                    issue = RecipeIssue(
                        (*group_path, key),
                        f"unknown value '{name}'. Not a dataset of this "
                        "variable group." + _suggest(name, datasets),
                        severity="warning",
                    )
                    issues.append(issue)
    return issues


def _find_missing_facets(
    recipe: object,
    config: Config | Session,
) -> list[RecipeIssue]:
    """Find datasets without the facets that their project requires.

    The required facets are configured with the ``required_facets`` option
    of the project. Facets that may come from the extra facets of the
    project are not required in the recipe.
    """
    issues = []
    for path, facets, _ in _datasets(recipe):
        project = facets.get("project")
        if not isinstance(project, str):
            continue
        settings = config["projects"].get(project, {})
        required = settings.get("required_facets", [])
        missing = [f for f in required if f not in facets]
        if missing:
            extra = _extra_facet_names(project, facets, config)
            missing = [f for f in missing if f not in extra]
        if missing:
            keys = "key" if len(missing) == 1 else "keys"
            names = ", ".join(repr(f) for f in missing)
            issue = RecipeIssue(
                path,
                f"missing required {keys} {names}. Needed to find the data "
                f"of project '{project}'.",
            )
            if issue not in issues:
                issues.append(issue)
    return issues


def _find_unknown_types(recipe: object) -> list[RecipeIssue]:
    """Find datasets with a type that their project does not use."""
    issues = []
    for _, facets, origin in _datasets(recipe):
        project = facets.get("project")
        if not isinstance(project, str) or project not in _TYPES:
            continue
        values = facets.get("type")
        values = values if isinstance(values, list) else [values]
        choices = _TYPES[project]
        listed = _join_or([repr(c) for c in choices])
        for value in values:
            if not isinstance(value, str) or value in choices:
                continue
            issue = RecipeIssue(
                (*origin["type"], "type"),
                f"unknown value {value!r}. Not a type of project "
                f"'{project}', choose from {listed}."
                + _suggest(value, choices),
                severity="warning",
            )
            if issue not in issues:
                issues.append(issue)
    return issues


def _find_invalid_timeranges(recipe: object) -> list[RecipeIssue]:
    """Find time ranges that are not valid.

    Time ranges with wildcards are only complete once the data has been
    found, so they are not checked.
    """
    issues = []
    for _, facets, origin in _datasets(recipe):
        timerange = facets.get("timerange")
        if not isinstance(timerange, str) or "*" in timerange:
            continue
        try:
            check.valid_time_selection(timerange)
        except RecipeError as exc:
            reason = str(exc).removeprefix(
                "Invalid value encountered for `timerange`. ",
            )
            issue = RecipeIssue(
                (*origin["timerange"], "timerange"),
                f"invalid value {timerange!r}. {reason}",
            )
            if issue not in issues:
                issues.append(issue)
    return issues


def _find_unknown_preprocessors(recipe: object) -> list[RecipeIssue]:
    """Find datasets that use a preprocessor that is not defined."""
    preprocessors = {"default", *map(str, _mapping(recipe, "preprocessors"))}
    return [
        RecipeIssue(
            path,
            f"unknown value {name!r}. Not defined in the `preprocessors` "
            "section." + _suggest(str(name), preprocessors),
        )
        for path, name in _preprocessor_facets(recipe)
        # The preprocessor is converted to a string when running the recipe.
        if str(name) not in preprocessors
    ]


def _suggest_task(pattern: str, tasks: list[str]) -> str:
    """Suggest a task of the same diagnostic with a similar name."""
    if any(char in pattern for char in "*?["):
        return ""
    diagnostic, _, name = pattern.rpartition(TASKSEP)
    names = {
        task.rpartition(TASKSEP)[2]: task
        for task in tasks
        if task.rpartition(TASKSEP)[0] == diagnostic
    }
    close = difflib.get_close_matches(name, list(names), n=1)
    return f" Did you mean '{names[close[0]]}'?" if close else ""


def _find_unknown_ancestors(recipe: object) -> list[RecipeIssue]:
    """Find ancestors of scripts that do not match any task."""
    diagnostics = _mapping(recipe, "diagnostics")
    tasks = [
        f"{diagnostic_name}{TASKSEP}{task}"
        for diagnostic_name, diagnostic in diagnostics.items()
        for group in ("variables", "scripts")
        for task in _mapping(diagnostic, group)
    ]
    issues = []
    for diagnostic_name, diagnostic in diagnostics.items():
        for script_name, script in _mapping(diagnostic, "scripts").items():
            ancestors = (
                script.get("ancestors") if isinstance(script, dict) else None
            )
            if not isinstance(ancestors, list):
                continue
            for i, ancestor in enumerate(ancestors):
                if not isinstance(ancestor, str):
                    continue
                # The same rules as in Recipe._initialize_scripts.
                pattern = (
                    ancestor
                    if TASKSEP in ancestor
                    else f"{diagnostic_name}{TASKSEP}{ancestor}"
                )
                if not fnmatch.filter(tasks, pattern):
                    issues.append(
                        RecipeIssue(
                            (
                                "diagnostics",
                                diagnostic_name,
                                "scripts",
                                script_name,
                                "ancestors",
                                i,
                            ),
                            f"unknown value {ancestor!r}. Does not match any "
                            "variable group or script."
                            + _suggest_task(pattern, tasks),
                        ),
                    )
    return issues


def _find_scripts_named_like_variables(recipe: object) -> list[RecipeIssue]:
    """Find scripts with the same name as a variable group."""
    return [
        RecipeIssue(
            ("diagnostics", name, "scripts", script),
            "a script cannot have the same name as a variable group of its "
            "diagnostic.",
        )
        for name, diagnostic in _mapping(recipe, "diagnostics").items()
        for script in _mapping(diagnostic, "scripts")
        if script in _mapping(diagnostic, "variables")
    ]


def _aggregator_issue(
    path: tuple[str | int, ...],
    operator: object,
    kwargs: dict,
) -> list[RecipeIssue]:
    """Check that the arguments of a statistics operator are valid."""
    # Unknown operators are reported by the schema.
    if operator not in STATISTICS_OPERATORS:
        return []
    try:
        get_iris_aggregator(str(operator), **kwargs)
    except ValueError as exc:
        return [RecipeIssue(path, f"{exc}.")]
    return []


def _statistics_issues(
    path: tuple[str | int, ...],
    step: str,
    settings: dict,
) -> list[RecipeIssue]:
    """Check the operators of a statistics step."""
    if step in ("multi_model_statistics", "ensemble_statistics"):
        statistics = settings.get("statistics")
        issues = []
        for i, statistic in enumerate(
            statistics if isinstance(statistics, list) else [],
        ):
            if isinstance(statistic, dict):
                kwargs = dict(statistic)
                operator = kwargs.pop("operator", None)
            else:
                operator, kwargs = statistic, {}
            issues.extend(
                _aggregator_issue((*path, "statistics", i), operator, kwargs),
            )
        return issues
    function = getattr(esmvalcore.preprocessor, step, None)
    if function is None:
        # Unknown preprocessor functions are reported by the schema.
        return []
    # The other arguments of the function, e.g. `hours` of
    # `hourly_statistics`, are not arguments of the operator.
    arguments = list(inspect.signature(function).parameters)[1:]
    kwargs = {k: v for k, v in settings.items() if k not in arguments}
    return _aggregator_issue(path, settings.get("operator", "mean"), kwargs)


def _find_invalid_step_settings(recipe: object) -> list[RecipeIssue]:
    """Find statistics operators and regridding schemes that do not work.

    These settings are passed on to iris or imported, so the schema cannot
    describe them.
    """
    issues = []
    for name, preprocessor in _mapping(recipe, "preprocessors").items():
        if not isinstance(preprocessor, dict):
            continue
        for step, settings in preprocessor.items():
            if not isinstance(settings, dict):
                continue
            path = ("preprocessors", name, step)
            if "_statistics" in step:
                issues.extend(_statistics_issues(path, step, settings))
            scheme = settings.get("scheme")
            if step == "regrid" and isinstance(scheme, dict):
                try:
                    _load_generic_scheme(scheme)
                except ValueError as exc:
                    url = _documentation_url(
                        "recipe/preprocessor.html#generic-regridding-schemes",
                    )
                    issues.append(
                        RecipeIssue(
                            (*path, "scheme"),
                            "failed to load generic regridding scheme: "
                            f"{exc} Refer to {url} for details.",
                        ),
                    )
    return issues


def _locate(
    issues: list[RecipeIssue],
    root: yaml.Node,
    aliases: _Aliases,
) -> list[RecipeIssue]:
    """Add line and column numbers to issues and sort them by position."""
    located = []
    for issue in issues:
        position = (
            None
            if issue.line is not None
            else _find_position(root, aliases, issue.path)
        )
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


class _NotGiven:
    """Marker for an argument that was not given."""

    def __repr__(self) -> str:
        return "<not given>"


_NOT_GIVEN = _NotGiven()


def validate_recipe(
    recipe: object = _NOT_GIVEN,
    text: str | None = None,
    session: Session | None = None,
) -> list[RecipeIssue]:
    """Validate a recipe against the JSON Schema.

    Also checks that every variable group has datasets, that no dataset occurs
    twice in a variable group, that datasets have a dataset, project, and MIP
    and the facets that their project requires, that the reference and
    alternative datasets of variable groups are datasets of the group, that the
    preprocessors of variable groups and datasets and the ancestors of scripts
    are defined in the recipe, that scripts and variable groups have different
    names, that time ranges without wildcards are valid, and that statistics
    operators and generic regridding schemes can be loaded.

    Parameters
    ----------
    recipe:
        The recipe, as loaded from YAML. If not given, it is loaded from
        ``text``.
    text:
        The YAML text of the recipe. If given, the issues include line and
        column numbers and are sorted by their position in the text, and
        keys that are defined more than once, merge keys (``<<``) that are
        used more than once or merge the mapping that contains them, and
        values that parsers of YAML 1.2 read differently are reported as
        warnings.
    session:
        The session that the recipe will be run with. Its configuration
        defines which projects, CMOR tables, required facets, and extra facets
        can be used. If not given, the configuration in
        :obj:`esmvalcore.config.CFG` is used.

    Returns
    -------
    :
        The problems found. The list is empty if the recipe is valid.

    Raises
    ------
    TypeError
        If neither ``recipe`` nor ``text`` is given.
    """
    if isinstance(recipe, _NotGiven):
        if text is None:
            msg = "Either `recipe` or `text` must be given."
            raise TypeError(msg)
        try:
            recipe = _parse_yaml(text)
        except yaml.YAMLError as exc:
            return [yaml_error_issue(exc)]
    return _validate(recipe, text, session, required_facets=True)


def _validate(
    recipe: object,
    text: str | None,
    session: Session | None,
    *,
    required_facets: bool,
) -> list[RecipeIssue]:
    """Validate a recipe, optionally without checking required facets.

    Datasets that do not have all facets needed to find their data are fine
    in the Python API, because the missing facets can be added later.
    """
    issues: list[RecipeIssue] = []
    config = CFG if session is None else session
    validator = jsonschema.Draft7Validator(generate_recipe_schema(session))
    for error in validator.iter_errors(_normalize(recipe)):
        for relevant, alternatives in _relevant_errors(error):
            for issue in _describe(relevant, alternatives):
                if issue not in issues:
                    issues.append(issue)
    # The schema also reports datasets without a dataset.
    issues.extend(
        issue
        for issue in _find_incomplete_datasets(recipe)
        if issue not in issues
    )
    if required_facets:
        issues.extend(_find_missing_facets(recipe, config))
    issues.extend(_find_unknown_types(recipe))
    issues.extend(_find_invalid_timeranges(recipe))
    issues.extend(_find_unknown_reference_datasets(recipe))
    issues.extend(_find_unknown_preprocessors(recipe))
    issues.extend(_find_unknown_ancestors(recipe))
    issues.extend(_find_scripts_named_like_variables(recipe))
    # Only report settings that do not work if the schema found no problem
    # with them, to avoid reporting the same problem twice.
    issues.extend(
        issue
        for issue in _find_invalid_step_settings(recipe)
        if not any(i.path[: len(issue.path)] == issue.path for i in issues)
    )
    composed = None if text is None else _compose(text)
    if composed is not None:
        checker = _YAMLChecker(composed)
        checker.check(composed.root, ())
        issues.extend(checker.issues)
        issues = _locate(issues, composed.root, composed.aliases)
    return issues


def ensure_valid_recipe(
    recipe: dict,
    path: Path,
    session: Session | None = None,
) -> None:
    """Validate a recipe, log the warnings, and raise an error if invalid."""
    logger.debug("Validating recipe %s", path)
    # The text is used to find duplicate keys and the line numbers of issues.
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        text = None
    issues = validate_recipe(recipe, text, session)
    for issue in issues:
        if issue.severity == "warning":
            logger.warning("Recipe %s: %s", path, issue)
    errors = [issue for issue in issues if issue.severity == "error"]
    if errors:
        msg = f"Recipe {path} is invalid:\n" + "\n".join(
            f"- {issue}" for issue in errors
        )
        raise RecipeError(msg)


def _is_about_datasets(issue: RecipeIssue) -> bool:
    """Check if a problem is in the datasets or variables of a recipe."""
    path = issue.path
    return path[:1] == ("datasets",) or (
        path[:1] == ("diagnostics",)
        and path[2:3] in (("additional_datasets",), ("variables",))
    )


def ensure_valid_datasets(
    recipe: dict,
    text: str | None,
    session: Session,
) -> None:
    """Validate the datasets and variables in a recipe.

    The recipe may be incomplete, so problems in other sections and facets
    that are needed to find the data but missing are ignored. Problems that
    prevent reading the datasets raise an error, other problems a deprecation
    warning.
    """
    issues = [
        issue
        for issue in _validate(recipe, text, session, required_facets=False)
        if _is_about_datasets(issue)
    ]
    for issue in issues:
        if issue.severity == "warning":
            logger.warning("Recipe: %s", issue)
    errors = [issue for issue in issues if issue.severity == "error"]
    fatal = {(i.path, i.message) for i in _find_incomplete_datasets(recipe)}
    if any((i.path, i.message) in fatal for i in errors):
        msg = "The datasets in the recipe are invalid:\n" + "\n".join(
            f"- {issue}" for issue in errors
        )
        raise RecipeError(msg)
    if errors:
        msg = (
            "The datasets in the recipe are invalid:\n"
            + "".join(f"- {issue}\n" for issue in errors)
            + "Reading datasets from an invalid recipe is deprecated in "
            "ESMValCore v2.16.0 and will raise an error in ESMValCore v2.18.0."
        )
        warnings.warn(msg, ESMValCoreDeprecationWarning, stacklevel=3)
