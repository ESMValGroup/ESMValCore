"""Test derivation of `cVeg_grid`."""
import mock

import esmvalcore.preprocessor._derive.cVeg_grid as cVeg_grid

CUBES = 'mocked cubes'
STD_NAME = 'vegetation_carbon_content'


@mock.patch.object(cVeg_grid, 'grid_area_correction', autospec=True)
def test_cveg_grid_calculation(mock_grid_area_correction):
    """Test calculation of `cVeg_grid."""
    derived_var = cVeg_grid.DerivedVariable()
    derived_var.calculate(CUBES)
    mock_grid_area_correction.assert_called_once_with(CUBES, STD_NAME)
