"""Integration test for
:func:`esmvalcore.preprocessor.regrid.get_reference_levels`."""
import iris.coords
import iris.cube
import iris.util
import numpy as np
import pytest

from esmvalcore.dataset import Dataset
from esmvalcore.preprocessor import _regrid


@pytest.fixture
def test_cube():
    cube = iris.cube.Cube(np.ones([2, 2, 2]), var_name='var')
    coord = iris.coords.DimCoord(np.arange(0, 2), var_name='coord')
    coord.attributes['positive'] = 'up'
    cube.add_dim_coord(coord, 0)
    return cube


def test_get_file_levels_from_coord(mocker, test_cube):
    dataset = mocker.create_autospec(Dataset, spec_set=True, instance=True)
    dataset.copy.return_value.load.return_value = test_cube
    reference_levels = _regrid.get_reference_levels(dataset)
    assert reference_levels == [0., 1]


def test_get_file_levels_from_coord_fail(mocker, test_cube):
    test_cube.coord('coord').attributes.clear()
    dataset = mocker.create_autospec(Dataset, spec_set=True, instance=True)
    dataset.copy.return_value.load.return_value = test_cube
    with pytest.raises(ValueError):
        _regrid.get_reference_levels(dataset)
