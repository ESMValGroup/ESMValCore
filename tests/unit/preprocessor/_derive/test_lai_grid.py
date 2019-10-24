"""Test derivation of `lai_grid`."""
import mock

import esmvalcore.preprocessor._derive.lai_grid as lai_grid

CUBES = 'mocked cubes'
VAR_NAME = 'lai'


@mock.patch.object(lai_grid, 'grid_area_correction', autospec=True)
def test_lai_grid_calculation(mock_grid_area_correction):
    """Test calculation of `lai_grid."""
    derived_var = lai_grid.DerivedVariable()
    derived_var.calculate(CUBES)
    mock_grid_area_correction.assert_called_once_with(CUBES, VAR_NAME)
