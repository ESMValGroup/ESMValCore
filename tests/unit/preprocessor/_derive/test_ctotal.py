"""Test derivation of `ctotal`."""
import iris
import numpy as np
import pytest
from cf_units import Unit

import esmvalcore.preprocessor._derive.ctotal as ctotal


@pytest.fixture
def cubes(project):
    if project == 'CMIP5':
        c_soil_cube = iris.cube.Cube([[1.0, 2.0], [0.0, 20.0]],
                                     units='kg m-2',
                                     standard_name='soil_carbon_content')
    elif project == 'CMIP6':
        c_soil_cube = iris.cube.Cube(
            [[1.0, 2.0], [0.0, 20.0]],
            units='kg m-2',
            standard_name='soil_mass_content_of_carbon')
    elif project == 'CMIPX':
        c_soil_cube = iris.cube.Cube([[1.0, 2.0], [0.0, 20.0]],
                                     units='kg m-2',
                                     standard_name='air_temperature')

    c_veg_cube = iris.cube.Cube([[10.0, 20.0], [50.0, 100.0]],
                                units='kg m-2',
                                standard_name='vegetation_carbon_content')
    return iris.cube.CubeList([c_soil_cube, c_veg_cube])


def _run_test(cubes):
    """Run the actual test on cubes list."""
    derived_var = ctotal.DerivedVariable()
    out_cube = derived_var.calculate(cubes)
    np.testing.assert_allclose(out_cube.data,
                               np.array([[11.0, 22.0], [50.0, 120.0]]))
    assert out_cube.units == Unit('kg m-2')


@pytest.mark.parametrize('project', ['CMIP5', 'CMIP6', 'CMIPX'])
def test_ctotal_calculation_cmip5(cubes, project):
    if project in ['CMIP5', 'CMIP6']:
        _run_test(cubes)
    else:
        msg = "soil_carbon_content or CMIP6: soil_mass_content_of_carbon"
        with pytest.raises(ValueError) as exp:
            _run_test(cubes)
        assert msg in str(exp.value)
