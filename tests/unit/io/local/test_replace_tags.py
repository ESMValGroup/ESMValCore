"""Tests for `_replace_tags` in `esmvalcore.io.local`."""

import re
from pathlib import Path

import pytest

from esmvalcore.io.local import (
    _MissingFacetError,
    _replace_tags,
)

VARIABLE = {
    "project": "CMIP6",
    "dataset": "ACCURATE-MODEL",
    "activity": "act",
    "exp": "experiment",
    "institute": "HMA",
    "ensemble": "r1i1p1f1",
    "mip": "Amon",
    "short_name": "tas",
    "grid": "gr",
}


def test_replace_tags():
    """Tests for `_replace_tags` function."""
    path = _replace_tags(
        "{activity}/{institute}/{dataset}/{exp}/{ensemble}/{mip}/{short_name}/"
        "{grid}/{version}",
        VARIABLE,
    )
    input_file = _replace_tags(
        "{short_name}_{mip}_{dataset}_{exp}_{ensemble}_{grid}*.nc",
        VARIABLE,
    )
    output_file = _replace_tags(
        "{project}_{dataset}_{mip}_{exp}_{ensemble}_{short_name}",
        VARIABLE,
    )
    assert path == [
        Path("act/HMA/ACCURATE-MODEL/experiment/r1i1p1f1/Amon/tas/gr/*"),
    ]
    assert input_file == [
        Path("tas_Amon_ACCURATE-MODEL_experiment_r1i1p1f1_gr*.nc"),
    ]
    assert output_file == [
        Path("CMIP6_ACCURATE-MODEL_Amon_experiment_r1i1p1f1_tas"),
    ]


def test_replace_tags_with_caps():
    """Test for `_replace_tags` function with .lower and .upper feature."""
    input_file = _replace_tags(
        "{short_name.upper}_{mip}_{dataset.lower}_{exp}_{ensemble}_{grid}*.nc",
        VARIABLE,
    )
    assert input_file == [
        Path("TAS_Amon_accurate-model_experiment_r1i1p1f1_gr*.nc"),
    ]


def test_replace_tags_missing_facet():
    """Check that a MissingFacetError is raised if a required facet is missing."""
    paths = ["{short_name}_{missing}_*.nc"]
    variable = {"short_name": "tas"}
    expected_message = (
        "Unable to complete path 'tas_{missing}_*.nc' because the facet "
        "'missing' has not been specified."
    )
    with pytest.raises(_MissingFacetError, match=re.escape(expected_message)):
        _replace_tags(paths, variable)


def test_replace_tags_missing_facets():
    """Check that a MissingFacetError is raised if multiple facets are missing."""
    paths = ["{missing1}_{short_name}_{missing2}_{missing3}_*.nc"]
    variable = {"short_name": "tas"}
    expected_message = (
        "Unable to complete path '{missing1}_tas_{missing2}_{missing3}_*.nc' "
        "because the facets 'missing1', 'missing2', and 'missing3' have not "
        "been specified."
    )
    with pytest.raises(_MissingFacetError, match=re.escape(expected_message)):
        _replace_tags(paths, variable)


def test_replace_tags_list_of_str():
    paths = [
        "folder/subfolder/{short_name}",
        "folder2/{short_name}",
        "subfolder/{short_name}",
    ]
    reference = [
        Path("folder/subfolder/tas"),
        Path("folder2/tas"),
        Path("subfolder/tas"),
    ]
    assert sorted(_replace_tags(paths, VARIABLE)) == reference


def test_replace_tags_with_subexperiment():
    """Tests for `_replace_tags` function."""
    variable = {"sub_experiment": "199411", **VARIABLE}
    paths = _replace_tags(
        "{activity}/{institute}/{dataset}/{exp}/{ensemble}/{mip}/{short_name}/"
        "{grid}/{version}",
        variable,
    )
    input_file = _replace_tags(
        "{short_name}_{mip}_{dataset}_{exp}_{ensemble}_{grid}*.nc",
        variable,
    )
    output_file = _replace_tags(
        "{project}_{dataset}_{mip}_{exp}_{ensemble}_{short_name}",
        variable,
    )
    expected_paths = [
        Path(
            "act/HMA/ACCURATE-MODEL/experiment/199411-r1i1p1f1/Amon/tas/gr/*",
        ),
        Path("act/HMA/ACCURATE-MODEL/experiment/r1i1p1f1/Amon/tas/gr/*"),
    ]
    assert sorted(paths) == expected_paths
    assert input_file == [
        Path("tas_Amon_ACCURATE-MODEL_experiment_199411-r1i1p1f1_gr*.nc"),
    ]
    assert output_file == [
        Path("CMIP6_ACCURATE-MODEL_Amon_experiment_199411-r1i1p1f1_tas"),
    ]
