import textwrap

from esmvalcore.experimental import recipe_output


def test_diagnostic_output_repr(mocker):
    """Test `DiagnosticOutput.__repr__`."""
    tasks = [
        mocker.create_autospec(recipe_output.TaskOutput, instance=True),
        mocker.create_autospec(recipe_output.TaskOutput, instance=True),
    ]
    for i, task in enumerate(tasks):
        task.__str__.return_value = f'Task-{i}'

    diagnostic = recipe_output.DiagnosticOutput(
        name='diagnostic_name',
        title='This is a diagnostic',
        description='With a description',
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


def test_recipe_output_add_to_filters(mocker):
    """Coverage test for `RecipeOutput._add_to_filters`."""

    output = recipe_output.RecipeOutput({})

    valid_attr = recipe_output.RecipeOutput.FILTER_ATTRS[0]
    output._add_to_filters({valid_attr: "single value"})
    output._add_to_filters(
        {valid_attr: ["list value 1", "repeated list value"]})
    output._add_to_filters(
        {valid_attr: ["list value 2", "repeated list value"]})

    assert len(output.filters) == 1
    assert valid_attr in output.filters
    assert len(output.filters[valid_attr]) == 4
    assert "single value" in output.filters[valid_attr]
    assert "list value 1" in output.filters[valid_attr]
    assert "list value 2" in output.filters[valid_attr]
    assert "repeated list value" in output.filters[valid_attr]


def test_recipe_output_add_to_filters_no_attributes(mocker):
    """Test `RecipeOutput._add_to_filters` with no attributes."""
    output = recipe_output.RecipeOutput({})
    output._add_to_filters({})
    assert len(output.filters) == 0


def test_recipe_output_add_to_filters_no_valid_attributes(mocker):
    """Test `RecipeOutput._add_to_filters` with no valid attributes."""
    output = recipe_output.RecipeOutput({})
    output._add_to_filters({"invalid_attribute": "value"})
    assert len(output.filters) == 0


def test_recipe_output_sort_filters(mocker):
    """Coverage test for `RecipeOutput._sort_filters`."""
    output = recipe_output.RecipeOutput({})
    valid_attr = recipe_output.RecipeOutput.FILTER_ATTRS[0]
    unsorted_attributes = ["1", "2", "4", "value", "3"]
    output._add_to_filters({valid_attr: unsorted_attributes})
    output._sort_filters
