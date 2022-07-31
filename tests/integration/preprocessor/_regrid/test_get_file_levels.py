"""Integration test for
:func:`esmvalcore.preprocessor.regrid.get_reference_levels`."""
import iris.coords
import iris.cube
import iris.util
import numpy as np

from esmvalcore.dataset import Dataset
from esmvalcore.preprocessor import _regrid


def test_get_file_levels_from_coord(mocker):
    cube = iris.cube.Cube(np.ones([2, 2, 2]), var_name='var')
    cube.add_dim_coord(iris.coords.DimCoord(np.arange(0, 2), var_name='coord'),
                       0)
    cube.coord('coord').attributes['positive'] = 'up'

    dataset = mocker.create_autospec(Dataset, spec_set=True, instance=True)
    dataset.copy.return_value.load.return_value = cube
    reference_levels = _regrid.get_reference_levels(dataset)
    assert reference_levels == [0., 1]
