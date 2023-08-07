"""
Remove this test and test file after iris fixes this
https://github.com/SciTools/iris/issues/5413
"""
from pathlib import Path

import iris
import pytest

from esmvalcore.preprocessor._area import extract_region


@pytest.fixture
def test_data_path():
    """Path to test data for CMOR fixes."""
    parent = Path(__file__).resolve().parent
    return parent


def test_extract_region_cell_ancil(test_data_path):
    """Test readding cell measures ancil variables after extract region."""
    cube_posix_path = test_data_path / "cube_for_intersection.nc"
    cube = iris.load_cube(cube_posix_path)

    # intersection cube loses cellmeas/ancillary variables
    # under normal (unpatched) conditions of extract_region
    ex1 = extract_region(cube,
                         start_longitude=-90,
                         end_longitude=40,
                         start_latitude=20,
                         end_latitude=80)

    # intersection cube doesn't lose cellmeas/ancillary variables
    # under normal (unpatched) conditions of extract_region
    # so duplication must be avoided
    ex2 = extract_region(cube,
                         start_longitude=160,
                         end_longitude=280,
                         start_latitude=-5,
                         end_latitude=5)

    expected_cm = cube.cell_measures()[0]
    result_cm = ex1.cell_measures()
    assert result_cm
    assert expected_cm.measure == result_cm[0].measure
    assert expected_cm.var_name == result_cm[0].var_name
    assert expected_cm.standard_name == result_cm[0].standard_name
    expected_ancil = cube.ancillary_variables()[0]
    result_ancil = ex1.ancillary_variables()
    assert result_ancil
    assert expected_ancil.var_name == result_ancil[0].var_name
    assert expected_ancil.standard_name == result_ancil[0].standard_name
    assert len(ex2.cell_measures()) == 1
    assert len(ex2.ancillary_variables()) == 1
