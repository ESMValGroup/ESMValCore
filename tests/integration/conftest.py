from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from esmvalcore.io.local import (
    LocalDataSource,
    LocalFile,
    _select_files,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from esmvalcore.typing import Facets, FacetValue


def _get_files(  # noqa: C901,PLR0912
    self: LocalDataSource,
    root_path: Path,
    facets: Facets,
    suffix: str = "nc",
) -> list[LocalFile]:
    """Return dummy files.

    Wildcards are only supported for `dataset` and `institute`; in this case
    return files for the two "models" AAA and BBB.

    """
    if facets["dataset"] == "*":
        all_facets = [
            {**facets, "dataset": "AAA", "institute": "A"},
            {**facets, "dataset": "BBB", "institute": "B"},
        ]
    else:
        all_facets = [facets]

    self.rootpath = root_path / "input"
    facets = dict(facets)
    if "original_short_name" in facets:
        facets["short_name"] = facets["original_short_name"]

    files = []
    for expanded_facets in all_facets:
        filenames = []
        filename = str(self._get_glob_patterns(**expanded_facets)[0])
        if filename.endswith("nc"):
            filename = f"{filename[:-2]}{suffix}"

        if filename.endswith(f"[_.]*{suffix}"):
            filename = filename.replace(f"[_.]*{suffix}", f"_*.{suffix}")

        if facets["frequency"] == "fx":
            intervals = [""]
        else:
            intervals = [
                "1990-1999",
                "2000-2009",
                "2010-2019",
            ]
        if filename.endswith(f"*.{suffix}"):
            filename = filename[: -len(f"*.{suffix}")]
            for interval in intervals:
                filenames.append(f"{filename}_{interval}.{suffix}")
        else:
            filenames.append(filename)

        for filename in filenames:
            file = LocalFile(filename)
            file.facets = dict(expanded_facets)
            if facets["frequency"] != "fx":
                for interval in intervals:
                    if interval in filename:
                        file.facets["timerange"] = interval.replace("-", "/")
            files.append(file)

    if "timerange" in facets:
        files = _select_files(files, facets["timerange"])

    return files


def _get_find_data_func(
    path: Path,
    suffix: str = "nc",
) -> Callable[..., list[LocalFile]]:
    def find_data(
        self: LocalDataSource,
        **facets: FacetValue,
    ) -> list[LocalFile]:
        return _get_files(self, path, facets, suffix)

    return find_data


@pytest.fixture
def patched_datafinder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    find_data = _get_find_data_func(tmp_path)
    monkeypatch.setattr(LocalDataSource, "find_data", find_data)


@pytest.fixture
def patched_datafinder_grib(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    find_data = _get_find_data_func(tmp_path, suffix="grib")
    monkeypatch.setattr(LocalDataSource, "find_data", find_data)


@pytest.fixture
def patched_failing_datafinder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failing data finder.

    Do not return files for:
    - fx files
    - Variable rsutcs for model AAA

    Otherwise, return files just like `patched_datafinder`.

    """

    def find_data(
        self: LocalDataSource,
        **facets: FacetValue,
    ) -> list[LocalFile]:
        files = _get_files(self, tmp_path, facets)
        if facets["frequency"] == "fx":
            files = []
        returned_files = []
        for file in files:
            if not ("AAA" in file.name and "rsutcs" in file.name):
                returned_files.append(file)
        return returned_files

    monkeypatch.setattr(LocalDataSource, "find_data", find_data)
