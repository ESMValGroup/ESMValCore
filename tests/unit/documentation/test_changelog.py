"""Tests for the changelog."""

import collections
import re
from pathlib import Path


def test_duplications_in_changelog():
    changelog_path = Path(__file__).parents[3].joinpath("doc", "changelog.rst")
    changelog = changelog_path.read_text(encoding="utf-8")

    # Find all pull requests
    pr_links = re.compile(
        "<https://github.com/ESMValGroup/ESMValCore/pull/[0-9]+>",
    )
    links = pr_links.findall(changelog)

    # Check for duplicates
    if len(links) != len(set(links)):
        print("The following PR are duplicated in the changelog:")
        print(
            "\n".join(
                (
                    link
                    for link, count in collections.Counter(links).items()
                    if count > 1
                ),
            ),
        )
        raise AssertionError
