"""Test derivation of `cSoil_grid`."""
import mock

from esmvalcore.preprocessor._derive import csoil_grid

CUBES = 'mocked cubes'
VAR_NAME = 'cSoil'


@mock.patch.object(csoil_grid, 'grid_area_correction', autospec=True)
def test_csoil_grid_calculation(mock_grid_area_correction):
    """Test calculation of `cSoil_grid."""
    derived_var = csoil_grid.DerivedVariable()
    derived_var.calculate(CUBES)
    mock_grid_area_correction.assert_called_once_with(CUBES, VAR_NAME)
