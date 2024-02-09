"""
Temporary test.

Remove this test and test file after iris fixes this
https://github.com/SciTools/iris/issues/5413 .
"""
import iris
import numpy as np
from cf_units import Unit
from iris.fileformats.pp import EARTH_RADIUS

from esmvalcore.preprocessor._area import extract_region
from esmvalcore.preprocessor._shared import guess_bounds


def make_cube():
    """Make a realistic cube with cell measure and ancil var."""
    coord_sys = iris.coord_systems.GeogCS(EARTH_RADIUS)
    data = np.ones((10, 192, 288), dtype=np.float32)
    time = iris.coords.DimCoord(
        np.arange(0., 10., 1.),
        standard_name='time',
        units=Unit('days since 1950-01-01 00:00:00',
                   calendar='360_day'))
    lons = iris.coords.DimCoord(
        [i + .5 for i in range(288)],
        standard_name='longitude',
        bounds=[[i, i + 1.] for i in range(288)],
        units='degrees_east',
        coord_system=coord_sys)
    lats = iris.coords.DimCoord(
        [i + .5 for i in range(192)],
        standard_name='latitude',
        bounds=[[i, i + 1.] for i in range(192)],
        units='degrees_north',
        coord_system=coord_sys,
    )
    coords_spec = [(time, 0), (lats, 1), (lons, 2)]
    simple_cube = iris.cube.Cube(data, dim_coords_and_dims=coords_spec)

    # add a cell measure
    simple_cube = guess_bounds(simple_cube, ['longitude', 'latitude'])
    grid_areas = iris.analysis.cartography.area_weights(simple_cube)
    measure = iris.coords.CellMeasure(
        grid_areas,
        standard_name='cell_area',
        units='m2',
        measure='area')
    simple_cube.add_cell_measure(measure, range(0, measure.ndim))

    # add ancillary variable
    ancillary_var = iris.coords.AncillaryVariable(
        simple_cube.data,
        standard_name='land_ice_area_fraction',
        var_name='sftgif',
        units='%')
    simple_cube.add_ancillary_variable(ancillary_var,
                                       range(0, simple_cube.ndim))

    return simple_cube


def test_extract_region_cell_ancil():
    """Test re-adding cell measures ancil variables after extract region."""
    cube = make_cube()

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
    np.testing.assert_array_equal(result_cm[0].shape, (10, 60, 58))
    assert expected_cm.standard_name == result_cm[0].standard_name
    expected_ancil = cube.ancillary_variables()[0]
    result_ancil = ex1.ancillary_variables()
    assert result_ancil
    assert expected_ancil.var_name == result_ancil[0].var_name
    assert expected_ancil.standard_name == result_ancil[0].standard_name
    np.testing.assert_array_equal(result_ancil[0].shape, (10, 60, 58))
    assert len(ex2.cell_measures()) == 1
    assert len(ex2.ancillary_variables()) == 1
