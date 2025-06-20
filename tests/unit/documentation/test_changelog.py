"""Tests for the changelog."""

import collections
import os
import re


def test_duplications_in_changelog():
    changelog_path = os.path.join(
        os.path.dirname(__file__),
        "../../..",
        "doc/changelog.rst",
    )
    with open(changelog_path, encoding="utf-8") as changelog:
        changelog = changelog.read()

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
