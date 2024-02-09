"""Unit tests for :mod:`esmvalcore.cmor._utils`."""

import pytest
from iris.cube import Cube

from esmvalcore.cmor._utils import _get_single_cube


@pytest.mark.parametrize(
    'cubes', [[Cube(0)], [Cube(0, var_name='x')], [Cube(0, var_name='y')]]
)
def test_get_single_cube_one_cube(cubes, caplog):
    """Test ``_get_single_cube``."""
    single_cube = _get_single_cube(cubes, 'x')
    assert single_cube == cubes[0]
    assert not caplog.records


@pytest.mark.parametrize(
    'dataset_str,msg', [
        (None, "Found variable x, but"),
        ('XYZ', "Found variable x in XYZ, but"),
    ]
)
@pytest.mark.parametrize(
    'cubes', [
        [Cube(0), Cube(0, var_name='x')],
        [Cube(0, var_name='x'), Cube(0)],
        [Cube(0, var_name='x'), Cube(0, var_name='x')],
        [Cube(0), Cube(0), Cube(0, var_name='x')],
    ]
)
def test_get_single_cube_multiple_cubes(cubes, dataset_str, msg, caplog):
    """Test ``_get_single_cube``."""
    single_cube = _get_single_cube(cubes, 'x', dataset_str=dataset_str)
    assert single_cube == Cube(0, var_name='x')
    assert len(caplog.records) == 1
    log = caplog.records[0]
    assert log.levelname == 'WARNING'
    assert msg in log.message


@pytest.mark.parametrize(
    'dataset_str,msg', [
        (None, "More than one cube found for variable x but"),
        ('XYZ', "More than one cube found for variable x in XYZ but"),
    ]
)
@pytest.mark.parametrize(
    'cubes', [
        [Cube(0), Cube(0)],
        [Cube(0, var_name='y'), Cube(0)],
        [Cube(0, var_name='y'), Cube(0, var_name='z')],
        [Cube(0), Cube(0), Cube(0, var_name='z')],
    ]
)
def test_get_single_cube_no_cubes_fail(cubes, dataset_str, msg):
    """Test ``_get_single_cube``."""
    with pytest.raises(ValueError, match=msg):
        _get_single_cube(cubes, 'x', dataset_str=dataset_str)
