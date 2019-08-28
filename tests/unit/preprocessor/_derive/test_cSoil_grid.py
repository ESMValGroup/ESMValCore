"""Test derivation of `cSoil_grid`."""
import mock

import esmvalcore.preprocessor._derive.cSoil_grid as cSoil_grid

CUBES = 'mocked cubes'
STD_NAME = 'soil_carbon_content'


@mock.patch.object(cSoil_grid, 'grid_area_correction', autospec=True)
def test_cSoil_grid_calculation(mock_grid_area_correction):
    """Test calculation of `cSoil_grid."""
    derived_var = cSoil_grid.DerivedVariable()
    derived_var.calculate(CUBES)
    mock_grid_area_correction.assert_called_once_with(CUBES, STD_NAME)
