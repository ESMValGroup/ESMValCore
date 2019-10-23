"""Test derivation of `nbp_grid`."""
import mock

import esmvalcore.preprocessor._derive.nbp_grid as nbp_grid

CUBES = 'mocked cubes'
VAR_NAME = 'nbp'


@mock.patch.object(nbp_grid, 'grid_area_correction', autospec=True)
def test_nbp_grid_calculation(mock_grid_area_correction):
    """Test calculation of `nbp_grid."""
    derived_var = nbp_grid.DerivedVariable()
    derived_var.calculate(CUBES)
    mock_grid_area_correction.assert_called_once_with(CUBES, VAR_NAME)
