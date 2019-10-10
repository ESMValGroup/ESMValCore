"""Test derivation of `lai_grid`."""
import mock

import esmvalcore.preprocessor._derive.lai_grid as lai_grid

CUBES = 'mocked cubes'
STD_NAME = 'leaf_area_index'


@mock.patch.object(lai_grid, 'grid_area_correction', autospec=True)
def test_lai_grid_calculation(mock_grid_area_correction):
    """Test calculation of `lai_grid."""
    derived_var = lai_grid.DerivedVariable()
    derived_var.calculate(CUBES)
    mock_grid_area_correction.assert_called_once_with(CUBES, STD_NAME)
