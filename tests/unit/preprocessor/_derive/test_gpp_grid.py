"""Test derivation of `gpp_grid`."""
import mock

import esmvalcore.preprocessor._derive.gpp_grid as gpp_grid

CUBES = 'mocked cubes'
VAR_NAME = 'gpp'


@mock.patch.object(gpp_grid, 'grid_area_correction', autospec=True)
def test_gpp_grid_calculation(mock_grid_area_correction):
    """Test calculation of `gpp_grid."""
    derived_var = gpp_grid.DerivedVariable()
    derived_var.calculate(CUBES)
    mock_grid_area_correction.assert_called_once_with(CUBES, VAR_NAME)
