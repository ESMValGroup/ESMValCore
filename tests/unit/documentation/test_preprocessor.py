"""Tests for the preprocessor documentation."""

import re
from pathlib import Path

from esmvalcore.preprocessor import DEFAULT_ORDER, INITIAL_STEPS


def test_preprocessor_documentation_labels() -> None:
    """Check that each preprocessor step that can be used in a recipe is documented.

    A step is documented if ``doc/recipe/preprocessor.rst`` has a label
    named after it, e.g. ``.. _extract_levels:``, or a section with its name
    as title, e.g. ````extract_levels````.
    """
    path = Path(__file__).parents[3] / "doc" / "recipe" / "preprocessor.rst"
    text = path.read_text(encoding="utf-8")
    titles = set(re.findall(r"^``(\w+)``\n[=\-~^*]{4,}$", text, re.MULTILINE))
    labels = set(re.findall(r"^\.\. _(\w+):$", text, re.MULTILINE))
    recipe_steps = set(DEFAULT_ORDER) - set(INITIAL_STEPS)
    assert recipe_steps - titles - labels == set()
