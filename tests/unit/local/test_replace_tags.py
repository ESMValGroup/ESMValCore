"""Tests for `_replace_tags` in `esmvalcore.local`."""
from pathlib import Path

import pytest

from esmvalcore.exceptions import RecipeError
from esmvalcore.local import _replace_tags

VARIABLE = {
    'project': 'CMIP6',
    'dataset': 'ACCURATE-MODEL',
    'activity': 'act',
    'exp': 'experiment',
    'institute': 'HMA',
    'ensemble': 'r1i1p1f1',
    'mip': 'Amon',
    'short_name': 'tas',
    'grid': 'gr',
}


def test_replace_tags():
    """Tests for `_replace_tags` function."""
    path = _replace_tags(
        '{activity}/{institute}/{dataset}/{exp}/{ensemble}/{mip}/{short_name}/'
        '{grid}/{version}', VARIABLE)
    input_file = _replace_tags(
        '{short_name}_{mip}_{dataset}_{exp}_{ensemble}_{grid}*.nc', VARIABLE)
    output_file = _replace_tags(
        '{project}_{dataset}_{mip}_{exp}_{ensemble}_{short_name}', VARIABLE)
    assert path == [
        Path('act/HMA/ACCURATE-MODEL/experiment/r1i1p1f1/Amon/tas/gr/*')
    ]
    assert input_file == [
        Path('tas_Amon_ACCURATE-MODEL_experiment_r1i1p1f1_gr*.nc')
    ]
    assert output_file == [
        Path('CMIP6_ACCURATE-MODEL_Amon_experiment_r1i1p1f1_tas')
    ]


def test_replace_tags_missing_facet():
    """Check that a RecipeError is raised if a required facet is missing."""
    paths = ['{short_name}_{missing}_*.nc']
    variable = {'short_name': 'tas'}
    with pytest.raises(RecipeError) as exc:
        _replace_tags(paths, variable)

    assert "Dataset key 'missing' must be specified" in exc.value.message


def test_replace_tags_list_of_str():
    paths = [
        'folder/subfolder/{short_name}',
        'folder2/{short_name}',
        'subfolder/{short_name}',
    ]
    reference = [
        Path('folder/subfolder/tas'),
        Path('folder2/tas'),
        Path('subfolder/tas'),
    ]
    assert sorted(_replace_tags(paths, VARIABLE)) == reference


def test_replace_tags_with_subexperiment():
    """Tests for `_replace_tags` function."""
    variable = {'sub_experiment': '199411', **VARIABLE}
    paths = _replace_tags(
        '{activity}/{institute}/{dataset}/{exp}/{ensemble}/{mip}/{short_name}/'
        '{grid}/{version}', variable)
    input_file = _replace_tags(
        '{short_name}_{mip}_{dataset}_{exp}_{ensemble}_{grid}*.nc', variable)
    output_file = _replace_tags(
        '{project}_{dataset}_{mip}_{exp}_{ensemble}_{short_name}', variable)
    expected_paths = [
        Path(
            'act/HMA/ACCURATE-MODEL/experiment/199411-r1i1p1f1/Amon/tas/gr/*'),
        Path('act/HMA/ACCURATE-MODEL/experiment/r1i1p1f1/Amon/tas/gr/*'),
    ]
    assert sorted(paths) == expected_paths
    assert input_file == [
        Path('tas_Amon_ACCURATE-MODEL_experiment_199411-r1i1p1f1_gr*.nc')
    ]
    assert output_file == [
        Path('CMIP6_ACCURATE-MODEL_Amon_experiment_199411-r1i1p1f1_tas')
    ]
