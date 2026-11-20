"""Write ESMValTool recipes in a standard YAML format."""

from __future__ import annotations

import bisect
import io
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable

# The document start marker and the comments and empty lines before it.
_HEADER = re.compile(
    r"(?:[ \t]*(?:#.*)?\r?\n)*---(?:[ \t]+#.*)?[ \t]*\r?\n",
)
_NULL_TAG = "tag:yaml.org,2002:null"
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


class RecipeDumper(yaml.SafeDumper):
    """YAML dumper for recipes that keeps the names of anchors and comments."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.anchor_names: dict[yaml.Node, str] = {}
        # The comments of the items, refer to _attach_comments.
        self.comments: dict[Hashable, _Comments] = {}
        self._item_comments: _Comments | None = None
        self._event_comments: dict[yaml.Event, _Comments] = {}
        # The comments to write after the current item at each depth.
        self._after: dict[int, list[_Comment]] = {}
        # The comments to write at the end of the current line.
        self._trailing: list[_Comment] = []

    def serialize_node(
        self,
        node: yaml.Node,
        parent: yaml.Node | None,
        index: int | yaml.Node | None,
    ) -> None:
        """Serialize a node and remember the comments of its item."""
        if isinstance(parent, yaml.SequenceNode):
            self._item_comments = self.comments.get((id(parent), index))
        elif isinstance(parent, yaml.MappingNode) and index is None:
            self._item_comments = self.comments.get(id(node))
        super().serialize_node(node, parent, index)

    def emit(self, event: yaml.Event) -> None:
        """Emit an event, with the comments of the item that it starts."""
        if self._item_comments is not None:
            self._event_comments[event] = self._item_comments
            self._item_comments = None
        super().emit(event)

    def _write_comment_lines(self, comments: list[_Comment]) -> None:
        for line in _comment_lines(comments):
            if line:
                self.write_indent()
                self.stream.write(line)
                self.column += len(line)
            else:
                self.write_line_break()
                self.write_line_break()

    def _write_comments(self) -> None:
        """Write the comments of the item that starts with this event."""
        depth = len(self.indents)
        self._write_comment_lines(self._after.pop(depth, []))
        comments = (
            None
            if self.event is None
            else self._event_comments.get(self.event)
        )
        if comments is None:
            return
        self._write_comment_lines(comments.before)
        if comments.after:
            self._after[depth] = comments.after
        if comments.trailing:
            self.write_indent()
            self._trailing = comments.trailing

    def expect_block_mapping_key(self, first: bool = False) -> None:
        """Write the comments of a mapping item and then the item."""
        self._write_comments()
        super().expect_block_mapping_key(first)

    def expect_block_sequence_item(self, first: bool = False) -> None:
        """Write the comments of a sequence item and then the item."""
        self._write_comments()
        super().expect_block_sequence_item(first)

    def process_scalar(self) -> None:
        """Write a scalar, without comments between its lines."""
        # The style has already been chosen in process_tag.
        if self.style in ("|", ">"):
            # A comment can follow the header of a block scalar.
            super().process_scalar()
        else:
            trailing, self._trailing = self._trailing, []
            super().process_scalar()
            self._trailing = trailing

    def write_line_break(self, data: str | None = None) -> None:
        """Write the comments at the end of the line and a line break."""
        if self._trailing:
            text = "".join(f"  {c.text}" for c in self._trailing)
            self._trailing = []
            self.stream.write(text)
            self.column += len(text)
        super().write_line_break(data)

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


_BREAKS = "\0\r\n\x85\u2028\u2029"


@dataclass
class _Comment:
    """A comment in the text of a recipe."""

    index: int
    line: int
    column: int
    text: str
    own_line: bool  # There is nothing but whitespace before it on its line.
    blank_before: bool  # The line before it is empty.


@dataclass
class _Comments:
    """The comments that belong to an item of a mapping or sequence."""

    before: list[_Comment] = field(default_factory=list)
    trailing: list[_Comment] = field(default_factory=list)
    after: list[_Comment] = field(default_factory=list)


@dataclass
class _Item:
    """An item of a mapping or sequence and where it starts in the text."""

    key: Hashable
    index: int
    line: int
    column: int
    container: yaml.MappingNode | yaml.SequenceNode
    parent: _Item | None


class _RecipeLoader(yaml.SafeLoader):
    """YAML loader that remembers the names of anchors and the comments."""

    def __init__(self, stream: str) -> None:
        super().__init__(stream)
        self.anchor_names: dict[yaml.Node, str] = {}
        self.comments: list[_Comment] = []
        # Where the sequence items start: at the "-" in block style.
        self.item_marks: dict[tuple[int, int], yaml.Mark] = {}
        self._entry_mark: yaml.Mark | None = None

    def _scan_comment(self) -> None:
        """Scan a comment, if there is one, and remember it."""
        if self.peek() != "#":
            return
        mark = self.get_mark()
        length = 0
        while self.peek(length) not in _BREAKS:
            length += 1
        start = self.pointer - mark.column
        before = self.buffer[start : self.pointer]
        previous = self.buffer[
            self.buffer.rfind("\n", 0, start - 1) + 1 : start
        ]
        self.comments.append(
            _Comment(
                index=mark.index,
                line=mark.line,
                column=mark.column,
                text=self.prefix(length).rstrip(),
                own_line=not before.strip(),
                blank_before=start > 0 and not previous.strip(),
            ),
        )
        self.forward(length)

    def scan_to_next_token(self) -> None:
        """Skip spaces, line breaks and comments, remembering the comments."""
        # Adapted from yaml.scanner.Scanner.scan_to_next_token.
        if self.index == 0 and self.peek() == "\ufeff":
            self.forward()
        while True:
            while self.peek() == " ":
                self.forward()
            self._scan_comment()
            if not self.scan_line_break():
                return
            if not self.flow_level:
                self.allow_simple_key = True

    def scan_block_scalar_ignored_line(self, start_mark: yaml.Mark) -> None:
        """Scan the rest of the header of a block scalar."""
        while self.peek() == " ":
            self.forward()
        self._scan_comment()
        super().scan_block_scalar_ignored_line(start_mark)

    def _remember_entry(self) -> None:
        """Remember where the "-" of the next block sequence item is."""
        if self.check_token(yaml.BlockEntryToken):
            self._entry_mark = self.peek_token().start_mark

    def parse_block_sequence_entry(self) -> yaml.Event:
        """Parse a block sequence item, remembering where its "-" is."""
        self._remember_entry()
        return super().parse_block_sequence_entry()

    def parse_indentless_sequence_entry(self) -> yaml.Event:
        """Parse a block sequence item, remembering where its "-" is."""
        self._remember_entry()
        return super().parse_indentless_sequence_entry()

    def compose_node(
        self,
        parent: yaml.Node | None,
        index: int,
    ) -> yaml.Node | None:
        """Compose a node and remember the name of its anchor."""
        event = self.peek_event()
        if isinstance(parent, yaml.SequenceNode):
            # Remember where sequence items start, also if they are aliases.
            mark = event.start_mark
            if not parent.flow_style and self._entry_mark is not None:
                mark = self._entry_mark
            self.item_marks[id(parent), index] = mark
        node = super().compose_node(parent, index)
        anchor = getattr(event, "anchor", None)
        is_alias = isinstance(event, yaml.AliasEvent)
        if node is not None and anchor is not None and not is_alias:
            self.anchor_names[node] = anchor
        return node


def _find_items(
    root: yaml.Node,
    item_marks: dict[tuple[int, int], yaml.Mark],
) -> list[_Item]:
    """Find the items of the mappings and sequences in document order."""
    items: list[_Item] = []
    seen: set[int] = set()

    def find(node: yaml.Node, parent: _Item | None) -> None:
        if id(node) in seen:
            return
        seen.add(id(node))
        if isinstance(node, yaml.MappingNode):
            for key_node, value_node in node.value:
                mark = key_node.start_mark
                item = _Item(
                    id(key_node),
                    mark.index,
                    mark.line,
                    mark.column,
                    node,
                    parent,
                )
                items.append(item)
                find(value_node, item)
        elif isinstance(node, yaml.SequenceNode):
            for number, value_node in enumerate(node.value):
                mark = item_marks[id(node), number]
                item = _Item(
                    (id(node), number),
                    mark.index,
                    mark.line,
                    mark.column,
                    node,
                    parent,
                )
                items.append(item)
                find(value_node, item)

    find(root, None)
    return items


def _attach_comments(
    items: list[_Item],
    comments: list[_Comment],
) -> tuple[dict[Hashable, _Comments], list[_Comment]]:
    """Find the item that each comment belongs to.

    A comment at the end of a line belongs to the last item that starts
    before it on that line. A comment on its own line belongs before the next
    item if it is not indented more than that item, otherwise after the item
    before it at the same indentation. Items in a collection that
    is written in flow style cannot have comments, so their comments belong
    to the closest item that is not in such a collection.

    Returns
    -------
    :
        The comments of each item and the comments before the first item.
    """
    starts = [item.index for item in items]
    result: dict[Hashable, _Comments] = {}
    leading = []
    for comment in comments:
        position = bisect.bisect_left(starts, comment.index)
        if position == 0:
            leading.append(comment)
            continue
        # The last (and deepest) item that starts before the comment and the
        # first (and outermost) item that starts after it.
        target = items[position - 1]
        following = items[position] if position < len(items) else None
        if not comment.own_line and target.line == comment.line:
            kind = "trailing"
        elif (
            comment.own_line
            and following is not None
            and comment.column <= following.column
        ):
            kind = "before"
            target = following
        else:
            kind = "after"
            while target.parent is not None and target.column > comment.column:
                target = target.parent
        # Everything inside a collection in flow style is in flow style.
        in_flow = None
        ancestor: _Item | None = target
        while ancestor is not None:
            if ancestor.container.flow_style:
                in_flow = ancestor
            ancestor = ancestor.parent
        moved = False
        if in_flow is not None and in_flow.parent is not None:
            target = in_flow.parent
            moved = True
        comments_of_target = result.setdefault(target.key, _Comments())
        if (
            kind == "trailing"
            and moved
            and (comments_of_target.trailing or comment.line != target.line)
        ):
            kind = "before"
        getattr(comments_of_target, kind).append(comment)
    return result, leading


def _comment_lines(comments: list[_Comment]) -> list[str]:
    """Get the lines of comments, with an empty line where there was one."""
    lines: list[str] = []
    for number, comment in enumerate(comments):
        if number and comment.blank_before:
            lines.append("")
        lines.append(comment.text)
    return lines


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
            self._mapping(node, _DOCUMENTATION)
        elif key == "datasets":
            self._datasets(node)
        elif key == "preprocessors":
            self._named(node, self._preprocessor)
        elif key == "diagnostics":
            self._named(node, self._diagnostic)
        else:
            self.generic(node)

    def _preprocessor(self, node: yaml.Node) -> None:
        # The order of the preprocessor steps may matter, so keep it.
        self._mapping(node, None, sort=False)

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
        if line.partition(":")[0] in top_level_keys:
            # Put the blank line before the comments above the item.
            start = len(formatted_lines)
            while start > 1 and (
                formatted_lines[start - 1].startswith("#")
                or (
                    not formatted_lines[start - 1]
                    and formatted_lines[start - 2].startswith("#")
                )
            ):
                start -= 1
            if formatted_lines[start - 1]:
                formatted_lines.insert(start, "")
        formatted_lines.append(line)

    return "\n".join(formatted_lines)


def _serialize(
    node: yaml.Node,
    anchor_names: dict[yaml.Node, str] | None = None,
    comments: dict[Hashable, _Comments] | None = None,
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
    dumper.comments = comments or {}
    try:
        dumper.open()
        dumper.serialize(node)
        dumper.close()
    finally:
        dumper.dispose()
    return add_blank_lines_between_top_level_items(stream.getvalue())


def _format_recipe_data(recipe: dict) -> str:
    """Format a recipe that has been loaded from YAML."""
    dumper = RecipeDumper(io.StringIO(), sort_keys=False)
    node = dumper.represent_data(_sort_datasets(recipe))
    _Formatter(sort=True).recipe(node)
    return f"{_serialize(node)}\n"


def format_recipe(recipe: str | dict) -> str:
    """Format a recipe in a standard way.

    The sections of the recipe and the facets of the datasets and variables
    are put in a standard order, and the style of the YAML is made
    consistent.

    If the recipe is given as YAML text, anchors, aliases, comments, and the
    document start marker ``---`` are kept. Comments move together with the
    item that they belong to. Comments inside a dataset that is written on
    one line are put at the end of that line or before the dataset. The order
    of the datasets, preprocessors, preprocessor steps, diagnostics, variable
    groups, and scripts is kept.

    If the recipe is given as a :obj:`dict`, the lists of datasets are
    sorted and the keys of other mappings are sorted by name, except the
    preprocessor steps.

    Parameters
    ----------
    recipe:
        The YAML text of the recipe or the recipe loaded from YAML.

    Returns
    -------
    :
        The formatted YAML text of the recipe.

    Raises
    ------
    yaml.YAMLError
        If the text is not valid YAML.
    """
    if isinstance(recipe, dict):
        return _format_recipe_data(recipe)
    text = recipe
    match = _HEADER.match(text)
    header = match.group() if match else ""
    loader = _RecipeLoader(text)
    try:
        node = loader.get_single_node()
    finally:
        loader.dispose()
    comments = [c for c in loader.comments if c.index >= len(header)]
    # A document with only a document start marker contains an empty null.
    if node is None or (node.tag == _NULL_TAG and not node.value):
        lines = _comment_lines(comments)
        return header + "".join(f"{line}\n" for line in [*lines, "{}"])
    items = _find_items(node, loader.item_marks)
    _Formatter(sort=False).recipe(node)
    attached, leading = _attach_comments(items, comments)
    lines = _comment_lines(leading)
    if leading and items and items[0].line > leading[-1].line + 1:
        lines.append("")
    body = _serialize(node, loader.anchor_names, attached)
    return header + "".join(f"{line}\n" for line in [*lines, body])
