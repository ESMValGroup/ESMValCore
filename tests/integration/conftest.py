import os
from collections.abc import Iterator
from pathlib import Path

import iris
import pytest

import esmvalcore.local
from esmvalcore.local import (
    LocalFile,
    _replace_tags,
    _select_drs,
    _select_files,
)
from esmvalcore.typing import Facets


def create_test_file(filename, tracking_id=None):
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    attributes = {}
    if tracking_id is not None:
        attributes["tracking_id"] = tracking_id
    cube = iris.cube.Cube([])
    cube.attributes.globals = attributes

    iris.save(cube, filename)


def _get_files(  # noqa: C901,PLR0912
    root_path: Path,
    facets: Facets,
    tracking_id: Iterator[int],
    suffix: str = "nc",
) -> tuple[list[LocalFile], list[Path]]:
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

    # Globs without expanded facets
    dir_template = _select_drs("input_dir", facets["project"], "default")  # type: ignore[arg-type]
    file_template = _select_drs("input_file", facets["project"], "default")  # type: ignore[arg-type]
    dir_globs = _replace_tags(dir_template, facets)
    file_globs = _replace_tags(file_template, facets)
    globs = sorted(
        root_path / "input" / d / f for d in dir_globs for f in file_globs
    )

    files = []
    for expanded_facets in all_facets:
        filenames = []
        dir_template = _select_drs(
            "input_dir",
            expanded_facets["project"],  # type: ignore[arg-type]
            "default",
        )
        file_template = _select_drs(
            "input_file",
            expanded_facets["project"],  # type: ignore[arg-type]
            "default",
        )

        dir_globs = _replace_tags(dir_template, expanded_facets)
        file_globs = _replace_tags(file_template, expanded_facets)
        filename = str(
            root_path / "input" / dir_globs[0] / Path(file_globs[0]).name,
        )
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

        if suffix == "nc":
            for filename in filenames:
                create_test_file(filename, next(tracking_id))

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

    return files, globs


def _tracking_ids(i=0):
    while True:
        yield i
        i += 1


def _get_find_files_func(path: Path, suffix: str = "nc"):
    tracking_id = _tracking_ids()

    def find_files(self, *, debug: bool = False, **facets):
        files, file_globs = _get_files(path, facets, tracking_id, suffix)
        if debug:
            return files, file_globs
        return files

    return find_files


@pytest.fixture
def patched_datafinder(tmp_path, monkeypatch):
    find_files = _get_find_files_func(tmp_path)
    monkeypatch.setattr(
        esmvalcore.local.LocalDataSource,
        "find_data",
        find_files,
    )


@pytest.fixture
def patched_datafinder_grib(tmp_path, monkeypatch):
    find_files = _get_find_files_func(tmp_path, suffix="grib")
    monkeypatch.setattr(
        esmvalcore.local.LocalDataSource,
        "find_data",
        find_files,
    )


@pytest.fixture
def patched_failing_datafinder(tmp_path, monkeypatch):
    """Failing data finder.

    Do not return files for:
    - fx files
    - Variable rsutcs for model AAA

    Otherwise, return files just like `patched_datafinder`.

    """
    tracking_id = _tracking_ids()

    def find_files(self, *, debug: bool = False, **facets):
        files, file_globs = _get_files(tmp_path, facets, tracking_id)
        if facets["frequency"] == "fx":
            files = []
        returned_files = []
        for file in files:
            if not ("AAA" in file.name and "rsutcs" in file.name):
                returned_files.append(file)
        if debug:
            return returned_files, file_globs
        return returned_files

    monkeypatch.setattr(
        esmvalcore.local.LocalDataSource,
        "find_data",
        find_files,
    )
