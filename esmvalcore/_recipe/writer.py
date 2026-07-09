"""Write ESMValTool recipes to YAML files."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    import os


class RecipeDumper(yaml.SafeDumper):
    """Custom YAML dumper for recipes."""

    def increase_indent(
        self,
        flow=False,
        indentless=False,  # noqa: ARG002
    ):
        """Increase indentation level."""
        # Increase the indentation level for lists that are values in a mapping.
        return super().increase_indent(flow, False)


class Recipe(dict):
    """Recipe."""


RecipeDumper.add_representer(
    Recipe,
    lambda dumper, data: dumper.represent_mapping(
        "tag:yaml.org,2002:map",
        {
            k: data[k]
            for k in sorted(
                data,
                key=lambda x: (
                    {
                        "documentation": 0,
                        "preprocessors": 2,
                        "datasets": 3,
                        "diagnostics": 4,
                    }.get(x, 1),
                    x,
                ),
            )
        }.items(),
        flow_style=False,
    ),
)


class RecipeDocumentation(dict):
    """Recipe documentation section."""


def strip(txt: str) -> str:
    """Strip leading and trailing whitespace from each line in a multi-line string."""
    return "\n".join(line.strip() for line in txt.splitlines())


RecipeDumper.add_representer(
    RecipeDocumentation,
    lambda dumper, data: dumper.represent_mapping(
        "tag:yaml.org,2002:map",
        {
            k: strip(data[k]) if k == "description" else data[k]
            for k in sorted(
                data,
                key=lambda x: (
                    {
                        "title": 0,
                        "description": 1,
                        "authors": 2,
                        "maintainer": 3,
                    }.get(x, 4),
                    x,
                ),
            )
        }.items(),
        flow_style=False,
    ),
)


class RecipePreprocessor(dict):
    """Preprocessor steps."""


RecipeDumper.add_representer(
    RecipePreprocessor,
    lambda dumper, data: dumper.represent_mapping(
        "tag:yaml.org,2002:map",
        data.items(),  # Prevents sorting.
        flow_style=False,
    ),
)


class RecipeDatasetList(list):
    """List of datasets."""


RecipeDumper.add_representer(
    RecipeDatasetList,
    lambda dumper, data: dumper.represent_sequence(
        "tag:yaml.org,2002:seq",
        sorted(
            data,
            key=lambda x: (
                x.get("project", ""),
                tuple(
                    (
                        k,
                        [int(i) if i else 0 for i in re.split(r"\D", x[k])]
                        if k == "ensemble" and isinstance(x[k], str)
                        else str(x[k]),
                    )
                    for k in sorted(x)
                ),
            ),
        ),
        flow_style=False,
    ),
)


class RecipeDataset(dict):
    """Dataset entry."""


RecipeDumper.add_representer(
    RecipeDataset,
    lambda dumper, data: dumper.represent_mapping(
        "tag:yaml.org,2002:map",
        {
            k: data[k]
            for k in sorted(
                data,
                key=lambda x: (
                    {
                        "start_year": 1,
                        "end_year": 2,
                        "supplementary_variables": 3,
                    }.get(x, 0),
                    x,
                ),
            )
        }.items(),
        flow_style=True,
    ),
)


class RecipeVariable(dict):
    """Variable entry."""


RecipeDumper.add_representer(
    RecipeVariable,
    lambda dumper, data: dumper.represent_mapping(
        "tag:yaml.org,2002:map",
        {
            k: data[k]
            for k in sorted(
                data,
                key=lambda x: (
                    {
                        "start_year": 1,
                        "end_year": 2,
                        "additional_datasets": 3,
                    }.get(x, 0),
                    x,
                ),
            )
        }.items(),
        flow_style=False,
    ),
)


class RecipeDiagnostic(dict):
    """Diagnostic entry."""


RecipeDumper.add_representer(
    RecipeDiagnostic,
    lambda dumper, data: dumper.represent_mapping(
        "tag:yaml.org,2002:map",
        {
            k: data[k]
            for k in sorted(
                data,
                key=lambda x: (
                    {
                        "description": 0,
                        "realms": 1,
                        "themes": 2,
                        "variables": 3,
                        "additional_datasets": 4,
                        "scripts": 5,
                    }.get(x, 6),
                    x,
                ),
            )
        }.items(),
        flow_style=False,
    ),
)


class RecipeScript(dict):
    """Diagnostic script entry."""


RecipeDumper.add_representer(
    RecipeScript,
    lambda dumper, data: dumper.represent_mapping(
        "tag:yaml.org,2002:map",
        {
            k: data[k] for k in sorted(data, key=lambda x: (x != "script", x))
        }.items(),
        flow_style=False,
    ),
)


def convert_to_recipe_objects(recipe: dict) -> Recipe:  # noqa: C901
    """Convert the recipe dictionary to Recipe objects for YAML representation.

    Parameters
    ----------
    recipe:
        The recipe to convert.

    Returns
    -------
    :
        The recipe with relevant elements replaced by Recipe* classes used
        to tell the YAML dumper how to write the recipe.
    """
    recipe = Recipe(recipe)
    if "documentation" in recipe:
        recipe["documentation"] = RecipeDocumentation(recipe["documentation"])
    if "datasets" in recipe:
        recipe["datasets"] = RecipeDatasetList(
            RecipeDataset(d) for d in recipe["datasets"]
        )
    if "preprocessors" in recipe:
        recipe["preprocessors"] = {
            k: RecipePreprocessor(v)
            for k, v in recipe["preprocessors"].items()
        }
    for diagnostic_name, diagnostic in recipe.get("diagnostics", {}).items():
        if "additional_datasets" in diagnostic:
            diagnostic["additional_datasets"] = RecipeDatasetList(
                RecipeDataset(d) for d in diagnostic["additional_datasets"]
            )
        for variable_group, variable in diagnostic.get(
            "variables",
            {},
        ).items():
            if variable is None:
                continue
            if "additional_datasets" in variable:
                variable["additional_datasets"] = RecipeDatasetList(
                    RecipeDataset(d) for d in variable["additional_datasets"]
                )
            recipe["diagnostics"][diagnostic_name]["variables"][
                variable_group
            ] = RecipeVariable(variable)
        if scripts := diagnostic.get("scripts"):
            for script_name, script in scripts.items():
                scripts[script_name] = RecipeScript(script)
        recipe["diagnostics"][diagnostic_name] = RecipeDiagnostic(diagnostic)
    return recipe


def add_blank_lines_between_top_level_items(yaml_text: str) -> str:
    """Insert blank lines between top-level YAML sections."""
    lines = yaml_text.splitlines()
    if not lines:
        return yaml_text

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


def to_yaml(
    recipe: dict,
    file: os.PathLike | None = None,
) -> str:
    """Convert a recipe to YAML format.

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
    recipe = convert_to_recipe_objects(recipe)
    txt = yaml.dump(
        recipe,
        Dumper=RecipeDumper,
        allow_unicode=True,
        width=200,
    )
    txt = add_blank_lines_between_top_level_items(txt)

    if file:
        recipe_file = Path(file)
        recipe_file.parent.mkdir(parents=True, exist_ok=True)
        recipe_file.write_text(f"{txt}\n", encoding="utf-8")

    return txt
