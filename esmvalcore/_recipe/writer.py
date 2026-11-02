"""Write ESMValTool recipes in a standard YAML format."""

from __future__ import annotations

import io
import re
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    import os
    from collections.abc import Callable

# The order of the keys in the sections of a recipe. Keys that are not
# listed get the default order of the section and are sorted by name.
_Order = tuple[dict[str, int], int]
_MERGE_TAG = "tag:yaml.org,2002:merge"
_NUMBER_TAGS = ("tag:yaml.org,2002:int", "tag:yaml.org,2002:float")
_TOP_LEVEL: _Order = (
    {"documentation": 0, "preprocessors": 2, "datasets": 3, "diagnostics": 4},
    1,
)
_DOCUMENTATION: _Order = (
    {"title": 0, "description": 1, "authors": 2, "maintainer": 3},
    4,
)
_DATASET: _Order = (
    {"start_year": 1, "end_year": 2, "supplementary_variables": 3},
    0,
)
_VARIABLE: _Order = (
    {"start_year": 1, "end_year": 2, "additional_datasets": 3},
    0,
)
_DIAGNOSTIC: _Order = (
    {
        "description": 0,
        "realms": 1,
        "themes": 2,
        "additional_datasets": 3,
        "variables": 4,
        "scripts": 5,
    },
    6,
)
_SCRIPT: _Order = ({"ancestors": 0, "script": 1}, 2)


# Width available for the text of the description (79 minus the indentation
# of the text of a block scalar in the documentation section).
_DESCRIPTION_WIDTH = 75
# Descriptions that are a single line of at most this length are not a block.
_SHORT_DESCRIPTION = 64
_LIST_ITEM = re.compile(r"[-*+]\s|\d+[.)]\s")


class RecipeDumper(yaml.SafeDumper):
    """YAML dumper for recipes that keeps the names of anchors."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.anchor_names: dict[yaml.Node, str] = {}

    def increase_indent(
        self,
        flow: bool = False,
        indentless: bool = False,  # noqa: ARG002
    ) -> None:
        """Increase indentation level."""
        # Increase the indentation level for lists that are values in a mapping.
        return super().increase_indent(flow, False)

    def anchor_node(self, node: yaml.Node) -> None:
        """Find the nodes that need an anchor, keeping the original anchors."""
        super().anchor_node(node)
        if node in self.anchor_names and self.anchors[node] is None:
            self.anchors[node] = self.generate_anchor(node)

    def generate_anchor(self, node: yaml.Node) -> str:
        """Use the original name of an anchor if there is one."""
        return self.anchor_names.get(node) or super().generate_anchor(node)


class _AnchorLoader(yaml.SafeLoader):
    """YAML loader that remembers the names of anchors."""

    def __init__(self, stream: str) -> None:
        super().__init__(stream)
        self.anchor_names: dict[yaml.Node, str] = {}

    def compose_node(
        self,
        parent: yaml.Node | None,
        index: int,
    ) -> yaml.Node | None:
        """Compose a node and remember the name of its anchor."""
        event = self.peek_event()
        node = super().compose_node(parent, index)
        anchor = getattr(event, "anchor", None)
        is_alias = isinstance(event, yaml.AliasEvent)
        if node is not None and anchor is not None and not is_alias:
            self.anchor_names[node] = anchor
        return node


def _key(item: tuple[yaml.Node, yaml.Node]) -> str:
    """Get the name of the key of a mapping item."""
    return str(item[0].value) if isinstance(item[0], yaml.ScalarNode) else ""


def _name_order(item: tuple[yaml.Node, yaml.Node]) -> tuple[float, str]:
    """Get the order of a mapping item by its key, with numbers first."""
    key = item[0]
    if isinstance(key, yaml.ScalarNode) and key.tag in _NUMBER_TAGS:
        try:
            return float(key.value.replace("_", "")), ""
        except ValueError:
            pass
    return float("inf"), _key(item)


def _normalize_description(text: str) -> str:
    """Join the lines of each paragraph of a description.

    Trailing whitespace, the indentation common to all lines, and blank lines
    at the start and end are removed, and runs of blank lines become one
    blank line. The lines of a paragraph are joined with single spaces.
    List items (starting with ``-``, ``*``, ``+``, ``1.`` or ``1)``) and
    lines that are more indented than the paragraph text stay on their own
    lines, keeping their relative indentation. A line that is more indented
    than the list item above it, and is not a list item itself, is joined
    to that item.
    """
    lines = [line.rstrip() for line in text.expandtabs().splitlines()]
    indents = [len(line) - len(line.lstrip()) for line in lines if line]
    base = min(indents, default=0)
    result: list[str] = []
    kind = ""  # Kind of the last output line: "", "text", "item" or "other".
    item_indent = 0
    for raw in lines:
        if not raw:
            if result and result[-1]:
                result.append("")
            kind = ""
            continue
        line = raw[base:]
        indent = len(line) - len(line.lstrip())
        content = line.lstrip()
        if _LIST_ITEM.match(content + " "):
            result.append(line)
            kind, item_indent = "item", indent
        elif (kind == "item" and indent > item_indent) or (
            kind == "text" and indent == 0
        ):
            result[-1] += f" {content}"
        else:
            result.append(line)
            kind = "text" if indent == 0 else "other"
    while result and not result[-1]:
        result.pop()
    return "\n".join(result)


def _wrap_description(text: str) -> str:
    """Wrap the lines of a normalized description.

    Paragraphs and list items are wrapped to ``_DESCRIPTION_WIDTH``, and
    the continuation lines of a list item are indented under its text. Lines
    that are more indented than the paragraph text and are not list items
    are not wrapped. A word that would start a continuation line and looks
    like a list item stays on the previous line, so that the result is
    normalized again to the same text.
    """
    result = []
    for line in text.split("\n"):
        content = line.lstrip()
        indent = line[: len(line) - len(content)]
        marker = _LIST_ITEM.match(content + " ")
        if marker:
            hanging = indent + " " * len(marker.group())
        elif indent or not content:
            result.append(line)
            continue
        else:
            hanging = ""
        wrapped = textwrap.wrap(
            line,
            width=_DESCRIPTION_WIDTH,
            subsequent_indent=hanging,
            break_long_words=False,
            break_on_hyphens=False,
        )
        for number, part in enumerate(wrapped):
            if number and _LIST_ITEM.match(part.lstrip() + " "):
                result[-1] += f" {part}"
            else:
                result.append(part)
    return "\n".join(result)


class _Formatter:
    """Put the keys of a recipe in a standard order and set their style.

    Parameters
    ----------
    sort:
        Sort the keys of mappings that have no standard order by name. If
        :obj:`False`, these keys stay in the same order, so that, for
        example, diagnostics and variable groups stay in the order in which
        they were written.
    """

    def __init__(self, *, sort: bool) -> None:
        self.sort = sort
        self._done: set[int] = set()

    def _first_visit(self, node: yaml.Node) -> bool:
        if id(node) in self._done:
            return False
        self._done.add(id(node))
        return True

    def _mapping(
        self,
        node: yaml.Node,
        order: _Order | None,
        visit: Callable[[str, yaml.Node], None] | None = None,
        *,
        flow: bool = False,
        sort: bool | None = None,
    ) -> None:
        """Format a mapping and the values in it."""
        if not isinstance(node, yaml.MappingNode):
            self.generic(node)
            return
        if not self._first_visit(node):
            return
        node.flow_style = flow
        if order is not None or (self.sort if sort is None else sort):
            ranks, default = order or ({}, 0)
            node.value.sort(
                key=lambda item: (
                    ranks.get(_key(item), default),
                    *_name_order(item),
                ),
            )
        for key_node, value_node in node.value:
            self.generic(key_node)
            if visit is None:
                self.generic(value_node)
            else:
                visit(_key((key_node, value_node)), value_node)
            if key_node.tag == _MERGE_TAG and isinstance(
                value_node,
                yaml.SequenceNode,
            ):
                # Write lists of merged mappings as [*first, *second].
                value_node.flow_style = True

    def _named(
        self,
        node: yaml.Node,
        visit: Callable[[yaml.Node], None],
    ) -> None:
        """Format a mapping of names, such as diagnostics, to their values."""
        self._mapping(node, None, lambda _, value: visit(value))

    def generic(self, node: yaml.Node) -> None:
        """Format a node that has no standard order."""
        if isinstance(node, yaml.ScalarNode):
            node.style = (
                "|"
                if node.tag == "tag:yaml.org,2002:str" and "\n" in node.value
                else None
            )
        elif isinstance(node, yaml.SequenceNode):
            if self._first_visit(node):
                node.flow_style = False
                for item in node.value:
                    self.generic(item)
        else:
            self._mapping(node, None)

    def recipe(self, node: yaml.Node) -> None:
        """Format a recipe."""
        self._mapping(node, _TOP_LEVEL, self._section)

    def _section(self, key: str, node: yaml.Node) -> None:
        if key == "documentation":
            self._mapping(node, _DOCUMENTATION, self._documentation)
        elif key == "datasets":
            self._datasets(node)
        elif key == "preprocessors":
            self._named(node, self._preprocessor)
        elif key == "diagnostics":
            self._named(node, self._diagnostic)
        else:
            self.generic(node)

    def _documentation(self, key: str, node: yaml.Node) -> None:
        if key == "description" and isinstance(node, yaml.ScalarNode):
            text = _normalize_description(node.value)
            node.value = _wrap_description(text)
            node.style = (
                None
                if "\n" not in node.value
                and len(node.value) <= _SHORT_DESCRIPTION
                else "|"
            )
            return
        self.generic(node)

    def _preprocessor(self, node: yaml.Node) -> None:
        # The order of the preprocessor steps may matter, so keep it.
        self._mapping(
            node,
            None,
            lambda _, value: self.generic(value),
            sort=False,
        )

    def _datasets(self, node: yaml.Node) -> None:
        if not isinstance(node, yaml.SequenceNode):
            self.generic(node)
            return
        if self._first_visit(node):
            node.flow_style = False
            for item in node.value:
                self._mapping(item, _DATASET, flow=True)

    def _diagnostic(self, node: yaml.Node) -> None:
        self._mapping(node, _DIAGNOSTIC, self._diagnostic_section)

    def _diagnostic_section(self, key: str, node: yaml.Node) -> None:
        if key == "additional_datasets":
            self._datasets(node)
        elif key == "variables":
            self._named(node, self._variable)
        elif key == "scripts":
            self._named(node, self._script)
        else:
            self.generic(node)

    def _variable(self, node: yaml.Node) -> None:
        self._mapping(node, _VARIABLE, self._variable_section)

    def _variable_section(self, key: str, node: yaml.Node) -> None:
        if key == "additional_datasets":
            self._datasets(node)
        else:
            self.generic(node)

    def _script(self, node: yaml.Node) -> None:
        self._mapping(node, _SCRIPT)


def _dataset_sort_key(dataset: dict) -> tuple:
    """Get the key for sorting datasets by project and facets."""
    return (
        dataset.get("project", ""),
        tuple(
            (
                k,
                [int(i) if i else 0 for i in re.split(r"\D", dataset[k])]
                if k == "ensemble" and isinstance(dataset[k], str)
                else str(dataset[k]),
            )
            for k in sorted(dataset)
        ),
    )


def _sort_datasets(recipe: dict) -> dict:
    """Sort the lists of datasets in a recipe."""

    def sort(data: dict, key: str) -> None:
        if isinstance(data, dict) and isinstance(data.get(key), list):
            data[key] = sorted(data[key], key=_dataset_sort_key)

    recipe = dict(recipe)
    sort(recipe, "datasets")
    diagnostics = recipe.get("diagnostics")
    if isinstance(diagnostics, dict):
        recipe["diagnostics"] = diagnostics = {
            name: dict(diag) if isinstance(diag, dict) else diag
            for name, diag in diagnostics.items()
        }
        for diagnostic in diagnostics.values():
            sort(diagnostic, "additional_datasets")
            variables = (
                diagnostic.get("variables")
                if isinstance(diagnostic, dict)
                else None
            )
            if isinstance(variables, dict):
                diagnostic["variables"] = variables = {
                    group: dict(var) if isinstance(var, dict) else var
                    for group, var in variables.items()
                }
                for variable in variables.values():
                    sort(variable, "additional_datasets")
    return recipe


def add_blank_lines_between_top_level_items(yaml_text: str) -> str:
    """Insert blank lines between top-level YAML sections."""
    lines = yaml_text.splitlines()

    top_level_keys = {
        "documentation",
        "preprocessors",
        "datasets",
        "diagnostics",
    }
    formatted_lines = [lines[0]]

    for line in lines[1:]:
        for key in top_level_keys:
            if line and line.startswith(key):
                formatted_lines.append("")
                break
        formatted_lines.append(line)

    return "\n".join(formatted_lines)


def _serialize(
    node: yaml.Node,
    anchor_names: dict[yaml.Node, str] | None = None,
) -> str:
    """Write a formatted recipe node as YAML."""
    stream = io.StringIO()
    dumper = RecipeDumper(
        stream,
        allow_unicode=True,
        width=200,
        default_flow_style=False,
    )
    dumper.anchor_names = anchor_names or {}
    try:
        dumper.open()
        dumper.serialize(node)
        dumper.close()
    finally:
        dumper.dispose()
    return add_blank_lines_between_top_level_items(stream.getvalue())


def to_yaml(
    recipe: dict,
    file: os.PathLike | None = None,
) -> str:
    """Convert a recipe to YAML format.

    The sections of the recipe and the facets of the datasets and variables
    are put in a standard order, the lists of datasets are sorted, and the
    keys of other mappings are sorted by name, except the preprocessor
    steps. The lines of each paragraph of the description of the recipe are
    joined and wrapped at 79 characters, keeping list items and indented
    lines on their own lines.

    Parameters
    ----------
    recipe:
        The recipe to write.
    file:
        If provided, the recipe will be written to this file.

    Returns
    -------
    :
        The YAML representation of the recipe.
    """
    dumper = RecipeDumper(io.StringIO(), sort_keys=False)
    node = dumper.represent_data(_sort_datasets(recipe))
    _Formatter(sort=True).recipe(node)
    txt = _serialize(node)

    if file:
        recipe_file = Path(file)
        recipe_file.parent.mkdir(parents=True, exist_ok=True)
        recipe_file.write_text(f"{txt}\n", encoding="utf-8")

    return txt


def format_recipe(text: str) -> str:
    """Format a recipe in a standard way.

    The sections of the recipe and the facets of the datasets and variables
    are put in a standard order, and the style of the YAML is made
    consistent. Anchors and aliases are kept, but comments are removed.
    The order of the datasets, preprocessors, preprocessor steps,
    diagnostics, variable groups, and scripts is kept. The lines of each
    paragraph of the description of the recipe are joined and wrapped at 79
    characters, keeping list items and indented lines on their own lines.

    Parameters
    ----------
    text:
        The YAML text of the recipe.

    Returns
    -------
    :
        The formatted YAML text of the recipe.

    Raises
    ------
    yaml.YAMLError
        If the text is not valid YAML.
    """
    loader = _AnchorLoader(text)
    try:
        node = loader.get_single_node()
    finally:
        loader.dispose()
    if node is None:
        return "{}\n"
    _Formatter(sort=False).recipe(node)
    return f"{_serialize(node, loader.anchor_names)}\n"
