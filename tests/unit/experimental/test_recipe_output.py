from __future__ import annotations

import re
import textwrap
from typing import TYPE_CHECKING

from esmvalcore.config._tasksep import TASKSEP
from esmvalcore.experimental import recipe_output
from esmvalcore.experimental.recipe_info import RecipeInfo

if TYPE_CHECKING:
    from esmvalcore.config import Session


def test_diagnostic_output_repr(mocker):
    """Test `DiagnosticOutput.__repr__`."""
    tasks = [
        mocker.create_autospec(recipe_output.TaskOutput, instance=True),
        mocker.create_autospec(recipe_output.TaskOutput, instance=True),
    ]
    for i, task in enumerate(tasks):
        task.__str__.return_value = f"Task-{i}"

    diagnostic = recipe_output.DiagnosticOutput(
        name="diagnostic_name",
        title="This is a diagnostic",
        description="With a description",
        task_output=tasks,
    )

    text = textwrap.dedent("""
        diagnostic_name:
          Task-0
          Task-1
        """).lstrip()

    assert repr(diagnostic) == text
    for task in tasks:
        task.__str__.assert_called_once()


def test_recipe_output_add_to_filters():
    """Coverage test for `RecipeOutput._add_to_filters`."""
    filters = {}
    valid_attr = recipe_output.RecipeOutput.FILTER_ATTRS[0]

    recipe_output.RecipeOutput._add_to_filters(
        filters,
        {valid_attr: "single value"},
    )
    recipe_output.RecipeOutput._add_to_filters(
        filters,
        {valid_attr: ["list value 1", "repeated list value"]},
    )
    recipe_output.RecipeOutput._add_to_filters(
        filters,
        {valid_attr: ["list value 2", "repeated list value"]},
    )

    assert len(filters) == 1
    assert valid_attr in filters
    assert len(filters[valid_attr]) == 4
    assert "single value" in filters[valid_attr]
    assert "list value 1" in filters[valid_attr]
    assert "list value 2" in filters[valid_attr]
    assert "repeated list value" in filters[valid_attr]


def test_recipe_output_add_to_filters_no_attributes():
    """Test `RecipeOutput._add_to_filters` with no attributes."""
    filters = {}
    recipe_output.RecipeOutput._add_to_filters(filters, {})
    assert len(filters) == 0


def test_recipe_output_add_to_filters_no_valid_attributes():
    """Test `RecipeOutput._add_to_filters` with no valid attributes."""
    filters = {}
    invalid = "invalid_attribute"
    recipe_output.RecipeOutput._add_to_filters(filters, {invalid: "value"})
    assert invalid not in recipe_output.RecipeOutput.FILTER_ATTRS
    assert len(filters) == 0


def test_recipe_output_sort_filters():
    """Coverage test for `RecipeOutput._sort_filters`."""
    filters = {}
    valid_attr = recipe_output.RecipeOutput.FILTER_ATTRS[0]
    unsorted_attributes = ["1", "2", "4", "value", "3"]
    recipe_output.RecipeOutput._add_to_filters(
        filters,
        {valid_attr: unsorted_attributes},
    )
    filters = recipe_output.RecipeOutput._sort_filters(filters)
    assert filters[valid_attr] == sorted(unsorted_attributes)


def test_write_html_with_csv_tables(session: Session) -> None:
    """Display CSV outputs with their metadata and existing output types."""
    session.work_dir.mkdir(parents=True)
    table_path = session.work_dir / "scores.csv"
    table_path.write_text(
        'Metric & unit,Value,Notes\n"Température, K",001,"line 1\n<b>line 2</b>"\n',
        encoding="utf-8",
    )
    empty_path = session.work_dir / "empty.csv"
    empty_path.touch()
    info = RecipeInfo(
        {
            "documentation": {"title": "Recipe", "description": "Example"},
            "diagnostics": {"tables": {}, "plots": {}},
        },
        filename="recipe.yml",
    )
    output = recipe_output.RecipeOutput(
        {
            f"tables{TASKSEP}scores": {
                str(table_path): {
                    "caption": "Model performance",
                    "realms": "ocean",
                },
                str(empty_path): {"caption": "Empty result"},
            },
            f"plots{TASKSEP}results": {
                str(session.plot_dir / "plot.png"): {"realms": "atmos"},
                str(session.work_dir / "data.nc"): {"caption": "Dataset"},
            },
        },
        session=session,
        info=info,
    )

    output.write_html()

    report = (session.session_dir / "index.html").read_text(encoding="utf-8")
    assert report.count("<table ") == 1
    assert '<th scope="col">Metric &amp; unit</th>' in report
    assert "<td>Température, K</td><td>001</td>" in report
    assert "<td>line 1\n&lt;b&gt;line 2&lt;/b&gt;</td>" in report
    assert re.search(r'class="div_figure[^"]*\bf_realms_ocean\b', report)
    assert "Model performance" in report
    assert "href='work/scores.csv'>download" in report
    assert "href='work/scores_citation.bibtex'>references" in report
    assert (
        "href='work/scores_data_citation_info.txt'>extra data citation"
        in report
    )
    assert "href='work/scores_provenance.xml'>provenance" in report
    assert "href='work/empty.csv'>download" in report
    assert "src='plots/plot.png'" in report
    assert "Data files <button" in report
    assert "href='work/data.nc'>download" in report
