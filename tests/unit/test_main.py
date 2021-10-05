import sys

import fire

from esmvalcore import _main
from esmvalcore.exceptions import RecipeError


def test_run_recipe_error(mocker, caplog):
    """Test a run of the tool with a mistake in the recipe."""
    program = mocker.patch.object(
        fire,
        'Fire',
        create_autospec=True,
        instance=True,
    )
    msg = "A mistake in the recipe"
    program.side_effect = RecipeError(msg)

    exit_ = mocker.patch.object(sys, 'exit', create_autspec=True)

    caplog.set_level('DEBUG')
    _main.run()
    print(caplog.text)

    # Check that the exit status is 1
    assert exit_.called_once_with(1)

    # Check that only the RecipeError is logged above DEBUG level
    errors = [r for r in caplog.records if r.levelname != 'DEBUG']
    assert len(errors) == 1
    assert errors[0].message == msg

    # Check that the stack trace is logged
    assert 'Traceback' in caplog.text
