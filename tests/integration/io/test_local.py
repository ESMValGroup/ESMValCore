"""Tests for `esmvalcore.io.local`."""

from __future__ import annotations

import os
import pprint
import warnings
from pathlib import Path

import pytest
import yaml

from esmvalcore.exceptions import ESMValCoreDeprecationWarning
from esmvalcore.io.local import (
    LocalDataSource,
    _parse_period,
)

# Load test configuration
with open(
    os.path.join(os.path.dirname(__file__), "data_finder.yml"),
    encoding="utf-8",
) as file:
    CONFIG = yaml.safe_load(file)


def print_path(path):
    """Print path."""
    txt = path
    if os.path.isdir(path):
        txt += "/"
    if os.path.islink(path):
        txt += " -> " + os.readlink(path)
    print(txt)


def tree(path):
    """Print path, similar to the the `tree` command."""
    print_path(path)
    for dirpath, dirnames, filenames in os.walk(path):
        for dirname in dirnames:
            print_path(os.path.join(dirpath, dirname))
        for filename in filenames:
            print_path(os.path.join(dirpath, filename))


def create_file(filename):
    """Create an empty file."""
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(filename, "a", encoding="utf-8"):
        pass


def create_tree(path, filenames=None, symlinks=None):
    """Create directory structure and files."""
    for filename in filenames or []:
        create_file(os.path.join(path, filename))

    for symlink in symlinks or []:
        link_name = os.path.join(path, symlink["link_name"])
        os.symlink(symlink["target"], link_name)


@pytest.fixture
def root(tmp_path):
    """Root function for tests."""
    dirname = str(tmp_path)
    yield dirname
    print("Directory structure was:")
    tree(dirname)


@pytest.mark.parametrize("cfg", CONFIG["get_input_filelist"])
def test_find_data(root, cfg):
    """Test retrieving input filelist."""
    data_source = LocalDataSource(
        name="test-data-source",
        project=cfg["variable"]["project"],
        rootpath=root,
        priority=1,
        dirname_template=cfg["dirname_template"],
        filename_template=cfg["filename_template"],
    )
    print(
        f"Testing {data_source} with variable:\n",
        pprint.pformat(cfg["variable"]),
    )
    create_tree(
        root,
        cfg.get("available_files"),
        cfg.get("available_symlinks"),
    )

    # Find files
    input_filelist = data_source.find_data(**cfg["variable"])
    # Test result
    ref_files = [Path(root, file) for file in cfg["found_files"]]
    ref_globs = [
        Path(root, d, f) for d in cfg["dirs"] for f in cfg["file_patterns"]
    ]
    assert [Path(f) for f in input_filelist] == sorted(ref_files)
    for pattern in ref_globs:
        assert str(pattern) in data_source.debug_info


# TODO: Remove in v2.18.0
@pytest.mark.parametrize(
    (
        "dirname_template",
        "filename_template",
        "var_type",
        "output_stream",
        "warning_raised",
        "found_file",
    ),
    [
        ("", "icon.nc", None, None, False, "icon.nc"),
        ("{var_type}", "icon.nc", "atm", None, False, "atm/icon.nc"),
        ("", "{var_type}.nc", "atm", None, False, "atm.nc"),
        ("{var_type}", "{var_type}.nc", "atm", None, False, "atm/atm.nc"),
        ("{var_type}", "icon.nc", None, "atm", True, "atm/icon.nc"),
        ("", "{var_type}.nc", None, "atm", True, "atm.nc"),
        ("{var_type}", "{var_type}.nc", None, "atm", True, "atm/atm.nc"),
        ("{output_stream}", "icon.nc", "lnd", "atm", False, "atm/icon.nc"),
        ("", "{output_stream}.nc", "lnd", "atm", False, "atm.nc"),
        (
            "{output_stream}",
            "{output_stream}.nc",
            "lnd",
            "atm",
            False,
            "atm/atm.nc",
        ),
        ("{output_stream}", "icon.nc", None, "atm", False, "atm/icon.nc"),
        ("", "{output_stream}.nc", None, "atm", False, "atm.nc"),
        (
            "{output_stream}",
            "{output_stream}.nc",
            None,
            "atm",
            False,
            "atm/atm.nc",
        ),
    ],
)
def test_find_data_icon_legacy_facets(
    root,
    dirname_template,
    filename_template,
    var_type,
    output_stream,
    warning_raised,
    found_file,
):
    data_source = LocalDataSource(
        name="test-icon-data-source",
        project="ICON",
        rootpath=root,
        priority=1,
        dirname_template=dirname_template,
        filename_template=filename_template,
    )
    variable = {
        "project": "ICON",
        "dataset": "ICON-XPP",
        "exp": "amip",
    }
    if var_type is not None:
        variable["var_type"] = var_type
    if output_stream is not None:
        variable["output_stream"] = output_stream

    create_tree(
        root,
        filenames=[
            "atm.nc",
            "icon.nc",
            "atm/atm.nc",
            "atm/icon.nc",
        ],
        symlinks=None,
    )

    if warning_raised:
        with pytest.warns(ESMValCoreDeprecationWarning):
            input_filelist = data_source.find_data(**variable)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            input_filelist = data_source.find_data(**variable)

    assert len(input_filelist) == 1
    assert Path(input_filelist[0]) == Path(root, found_file)


def test_find_data_facet_missing() -> None:
    """Test that a MissingFacetError is raised if a required facet is missing."""
    data_source = LocalDataSource(
        name="test-data-source",
        project="CMIP6",
        rootpath=Path("/data/cmip6"),
        priority=1,
        dirname_template="{dataset}/{exp}/{ensemble}",
        filename_template="{short_name}.nc",
    )
    facets = {
        "short_name": "tas",
        "dataset": "test-dataset",
        "exp": ["historical", "ssp585"],
    }
    expected_message = (
        "Unable to complete paths 'test-dataset/historical/{ensemble}' and "
        "'test-dataset/ssp585/{ensemble}' because the facet 'ensemble' has "
        "not been specified."
    )
    files = data_source.find_data(**facets)
    assert not files
    assert data_source.debug_info == expected_message


def test_parse_period_invalid_timerange_type():
    msg = r"`timerange` should be a `str`, got <class 'int'>"
    with pytest.raises(TypeError, match=msg):
        _parse_period(1)
