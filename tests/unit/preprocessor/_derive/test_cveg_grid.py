"""Test derivation of `cVeg_grid`."""
import mock

from esmvalcore.preprocessor._derive import cveg_grid

CUBES = 'mocked cubes'
VAR_NAME = 'cVeg'


@mock.patch.object(cveg_grid, 'grid_area_correction', autospec=True)
def test_cveg_grid_calculation(mock_grid_area_correction):
    """Test calculation of `cVeg_grid."""
    derived_var = cveg_grid.DerivedVariable()
    derived_var.calculate(CUBES)
    mock_grid_area_correction.assert_called_once_with(CUBES, VAR_NAME)
