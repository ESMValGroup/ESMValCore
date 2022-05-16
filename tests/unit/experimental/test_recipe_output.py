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
