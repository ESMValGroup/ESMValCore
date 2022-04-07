import pytest

from esmvalcore._main import parse_resume


def create_previous_run(path, suffix):
    """Create a mock previous run of the tool."""
    prev_run = path / f'recipe_test_{suffix}'
    prev_recipe = prev_run / 'run' / 'recipe_test.yml'
    prev_recipe.parent.mkdir(parents=True)
    prev_recipe.write_text('test')

    return prev_run


def test_parse_resume(tmp_path):
    """Test `esmvalcore._main.parse_resume`."""
    prev_run1 = create_previous_run(tmp_path, '20210923_112001')
    prev_run2 = create_previous_run(tmp_path, '20210924_123553')

    recipe = tmp_path / 'recipe_test.yml'
    recipe.write_text('test')

    resume_dirs = parse_resume(f"{prev_run1} {prev_run2}", recipe)
    assert resume_dirs == [prev_run1, prev_run2]


def test_noop(tmp_path):
    """Test `esmvalcore._main.parse_resume`.

    Test that not using the resume option works.
    """
    recipe = tmp_path / 'recipe_test.yml'
    resume_dirs = parse_resume(None, recipe)
    assert resume_dirs == []


def test_fail_on_different_recipe(tmp_path):
    """Test `esmvalcore._main.parse_resume`.

    Test that trying to resume a different recipe fails.
    """
    prev_run = create_previous_run(tmp_path, '20210924_123553')

    recipe = tmp_path / 'recipe_test.yml'
    recipe.write_text('something else')

    with pytest.raises(ValueError):
        parse_resume(str(prev_run), recipe)
