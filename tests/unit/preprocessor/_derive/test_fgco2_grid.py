"""Test derivation of `fgco2_grid`."""
import mock

import esmvalcore.preprocessor._derive.fgco2_grid as fgco2_grid

CUBES = 'mocked cubes'
VAR_NAME = 'fgco2'


@mock.patch.object(fgco2_grid, 'grid_area_correction', autospec=True)
def test_fgco2_grid_calculation(mock_grid_area_correction):
    """Test calculation of `fgco2_grid."""
    derived_var = fgco2_grid.DerivedVariable()
    derived_var.calculate(CUBES)
    mock_grid_area_correction.assert_called_once_with(
        CUBES, VAR_NAME, ocean_var=True)
