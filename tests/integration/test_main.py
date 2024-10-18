"""Tests for ESMValTool CLI.

Includes a context manager to temporarily modify sys.argv
"""

import contextlib
import copy
import functools
import os
import shutil
import sys
from pathlib import Path
from textwrap import dedent
from unittest.mock import patch

import pytest
from fire.core import FireExit

import esmvalcore.config._config
from esmvalcore._main import Config, ESMValTool, Recipes, run
from esmvalcore.exceptions import RecipeError


def wrapper(f):
    @functools.wraps(f)
    def empty(*args, **kwargs):
        if kwargs:
            raise ValueError(f"Parameters not supported: {kwargs}")
        return True

    return empty


@contextlib.contextmanager
def arguments(*args):
    backup = sys.argv
    sys.argv = list(args)
    try:
        yield
    finally:
        sys.argv = backup


def test_setargs():
    original = copy.deepcopy(sys.argv)
    with arguments("testing", "working", "with", "sys.argv"):
        assert sys.argv == ["testing", "working", "with", "sys.argv"]
    assert sys.argv == original


@patch("esmvalcore._main.ESMValTool.version", new=wrapper(ESMValTool.version))
def test_version():
    """Test version command."""
    with arguments("esmvaltool", "version"):
        run()
    with arguments("esmvaltool", "version", "--extra_parameter=asterisk"):
        with pytest.raises(FireExit):
            run()


@patch("esmvalcore._main.ESMValTool.run", new=wrapper(ESMValTool.run))
def test_run():
    """Test version command."""
    with arguments("esmvaltool", "run", "recipe.yml"):
        run()


def test_empty_run(tmp_path, monkeypatch):
    """Test real run with no diags."""
    monkeypatch.delitem(  # TODO: remove in v2.14.0
        esmvalcore.config.CFG._mapping, "config_file", raising=False
    )
    recipe_file = tmp_path / "recipe.yml"
    content = dedent("""
        documentation:
          title: Test recipe
          description: This is a test recipe.
          authors:
            - andela_bouwe
          references:
            - contact_authors
            - acknow_project
          projects:
            - c3s-magic
        diagnostics: null
    """)
    recipe_file.write_text(content)
    log_dir = f"{tmp_path}/esmvaltool_output"
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "config.yml"
    config_file.write_text(f"output_dir: {log_dir}")

    with pytest.raises(RecipeError) as exc:
        ESMValTool().run(recipe_file, config_dir=config_dir)
    assert str(exc.value) == "The given recipe does not have any diagnostic."
    log_file = os.path.join(
        log_dir, os.listdir(log_dir)[0], "run", "main_log.txt"
    )
    filled_recipe = os.path.exists(
        log_dir + "/" + os.listdir(log_dir)[0] + "/run/recipe_filled.yml"
    )
    shutil.rmtree(log_dir)

    assert log_file
    assert not filled_recipe


def test_recipes_get(tmp_path, monkeypatch):
    """Test version command."""
    src_recipe = tmp_path / "recipe.yml"
    src_recipe.touch()
    tgt_dir = tmp_path / "test"
    tgt_dir.mkdir()
    monkeypatch.chdir(tgt_dir)
    with arguments("esmvaltool", "recipes", "get", str(src_recipe)):
        run()
    assert (tgt_dir / "recipe.yml").is_file()


@patch("esmvalcore._main.Recipes.list", new=wrapper(Recipes.list))
def test_recipes_list():
    """Test version command."""
    with arguments("esmvaltool", "recipes", "list"):
        run()


@patch("esmvalcore._main.Recipes.list", new=wrapper(Recipes.list))
def test_recipes_list_do_not_admit_parameters():
    """Test version command."""
    with arguments("esmvaltool", "recipes", "list", "parameter"):
        with pytest.raises(FireExit):
            run()


@patch(
    "esmvalcore._main.Config.get_config_developer",
    new=wrapper(Config.get_config_developer),
)
def test_get_config_developer():
    """Test version command."""
    with arguments("esmvaltool", "config", "get_config_developer"):
        run()


def test_get_config_developer_no_path():
    """Test version command."""
    with arguments("esmvaltool", "config", "get_config_developer"):
        run()
    config_file = Path.home() / ".esmvaltool" / "config-developer.yml"
    assert config_file.is_file()


def test_get_config_developer_path(tmp_path):
    """Test version command."""
    new_path = tmp_path / "subdir"
    with arguments(
        "esmvaltool", "config", "get_config_developer", f"--path={new_path}"
    ):
        run()
    assert (new_path / "config-developer.yml").is_file()


def test_get_config_developer_overwrite(tmp_path):
    """Test version command."""
    config_developer = tmp_path / "config-developer.yml"
    config_developer.write_text("old text")
    with arguments(
        "esmvaltool",
        "config",
        "get_config_developer",
        f"--path={tmp_path}",
        "--overwrite",
    ):
        run()
    assert config_developer.read_text() != "old text"


def test_get_config_developer_no_overwrite(tmp_path):
    """Test version command."""
    config_developer = tmp_path / "configuration_file.yml"
    config_developer.write_text("old text")
    with arguments(
        "esmvaltool",
        "config",
        "get_config_developer",
        f"--path={config_developer}",
    ):
        run()
    assert config_developer.read_text() == "old text"


@patch(
    "esmvalcore._main.Config.get_config_developer",
    new=wrapper(Config.get_config_developer),
)
def test_get_config_developer_bad_option_fails():
    """Test version command."""
    with arguments(
        "esmvaltool", "config", "get_config_developer", "--bad_option=path"
    ):
        with pytest.raises(FireExit):
            run()


@patch(
    "esmvalcore._main.Config.get_config_user",
    new=wrapper(Config.get_config_user),
)
def test_get_config_user():
    """Test version command."""
    with arguments("esmvaltool", "config", "get_config_user"):
        run()


def test_get_config_user_no_path():
    """Test version command."""
    with arguments("esmvaltool", "config", "get_config_user"):
        run()
    config_file = Path.home() / ".config" / "esmvaltool" / "config-user.yml"
    assert config_file.is_file()


def test_get_config_user_path(tmp_path):
    """Test version command."""
    new_path = tmp_path / "subdir"
    with arguments(
        "esmvaltool", "config", "get_config_user", f"--path={new_path}"
    ):
        run()
    assert (new_path / "config-user.yml").is_file()


def test_get_config_user_overwrite(tmp_path):
    """Test version command."""
    config_user = tmp_path / "config-user.yml"
    config_user.write_text("old text")
    with arguments(
        "esmvaltool",
        "config",
        "get_config_user",
        f"--path={tmp_path}",
        "--overwrite",
    ):
        run()
    assert config_user.read_text() != "old text"


def test_get_config_user_no_overwrite(tmp_path):
    """Test version command."""
    config_user = tmp_path / "configuration_file.yml"
    config_user.write_text("old text")
    with arguments(
        "esmvaltool", "config", "get_config_user", f"--path={config_user}"
    ):
        run()
    assert config_user.read_text() == "old text"


@patch(
    "esmvalcore._main.Config.get_config_user",
    new=wrapper(Config.get_config_user),
)
def test_get_config_user_bad_option_fails():
    """Test version command."""
    with arguments(
        "esmvaltool", "config", "get_config_user", "--bad_option=path"
    ):
        with pytest.raises(FireExit):
            run()
