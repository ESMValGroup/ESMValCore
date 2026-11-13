"""Fixtures for the tests of recipes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import yaml

from esmvalcore._recipe import json_schema, validate
from esmvalcore.config._diagnostics import Diagnostics
from tests.unit.recipe.example_recipes import AUTHORS

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def esmvaltool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Diagnostics:
    """Pretend that ESMValTool is installed."""
    diagnostics = Diagnostics(tmp_path / "esmvaltool")
    diagnostics.references.mkdir(parents=True)
    (diagnostics.scripts / "examples").mkdir(parents=True)
    (diagnostics.scripts / "examples" / "diagnostic.py").write_text("")
    (diagnostics.references / "eyring16gmd.bibtex").write_text("")
    tags = {
        "authors": {a: {"name": a} for a in AUTHORS},
        "projects": {"c3s-magic": "C3S MAGIC project"},
        "realms": {"atmos": "atmosphere", "ocean": "ocean"},
        "themes": {"phys": "physics"},
    }
    diagnostics.tags_config.write_text(
        yaml.safe_dump(tags, sort_keys=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(json_schema, "DIAGNOSTICS", diagnostics)
    monkeypatch.setattr(validate, "DIAGNOSTICS", diagnostics)
    monkeypatch.setattr(json_schema, "TAGS", diagnostics.load_tags())
    return diagnostics
