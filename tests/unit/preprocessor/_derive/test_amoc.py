"""Test derivation of `amoc`."""
import iris
import iris.fileformats
import numpy as np
import pytest

import esmvalcore.preprocessor._derive.amoc as amoc

from .test_shared import get_cube


@pytest.fixture
def cubes():
    # standard names
    msftmyz_name = 'ocean_meridional_overturning_mass_streamfunction'
    msftyz_name = 'ocean_y_overturning_mass_streamfunction'

    msftmyz_cube = get_cube([[[[100.]], [[100.]], [[100.]]]],
                            air_pressure_coord=False,
                            depth_coord=True,
                            standard_name=msftmyz_name)
    msftyz_cube = get_cube([[[[100.]], [[100.]], [[100.]]]],
                           air_pressure_coord=False,
                           depth_coord=True,
                           standard_name=msftyz_name)
    rando_cube = get_cube([[[[100.]], [[100.]], [[100.]]]],
                          air_pressure_coord=False,
                          depth_coord=True,
                          standard_name="air_temperature")
    msftmyz_cube.coord("latitude").points = np.array([26.0])
    msftyz_cube.coord("latitude").points = np.array([26.0])
    msftyz_cube.coord("latitude").standard_name = "grid_latitude"

    return (
        iris.cube.CubeList([msftmyz_cube]),
        iris.cube.CubeList([msftyz_cube]),
        iris.cube.CubeList([rando_cube]),
    )


def test_amoc_preamble(cubes):
    derived_var = amoc.DerivedVariable()

    cmip5_required = derived_var.required("CMIP5")
    assert "msftmyz" == cmip5_required[0]["short_name"]
    cmip6_required = derived_var.required("CMIP6")
    assert "msftmz" == cmip6_required[0]["short_name"]
    assert "msftyz" == cmip6_required[1]["short_name"]

    # if project s neither CMIP5 nor CMIP6
    with pytest.raises(ValueError) as verr:
        derived_var.required("CMIPX")
        assert "Project CMIPX can not be used" in verr

    cmip5_cubes = cubes[0]
    cmip6_cubes = cubes[1]
    rando_cubes = cubes[2]

    # other amoc-specific exceptions returned
    with pytest.raises(ValueError) as verr:
        derived_var.calculate(cmip5_cubes)
        assert "doesn't contain Atlantic Region" in verr
    with pytest.raises(ValueError) as verr:
        derived_var.calculate(cmip6_cubes)
        assert "doesn't contain Atlantic Region" in verr
    with pytest.raises(iris.exceptions.ConstraintMismatchError) as verr:
        derived_var.calculate(rando_cubes)
        assert "standard names could not be found" in verr


def build_ocean_cube(std_name):
    """Test the actual calculation of the amoc."""
    # assemble a decent cube this time
    coord_sys = iris.coord_systems.GeogCS(
        iris.fileformats.pp.EARTH_RADIUS)
    data = np.ones((5, 180, 360, 3), dtype=np.float32)
    lons = iris.coords.DimCoord(
        range(0, 360),
        standard_name='longitude',
        bounds=None,
        units='degrees_east',
        coord_system=coord_sys)
    lats = iris.coords.DimCoord(
        range(-90, 90),
        standard_name='latitude',
        bounds=None,
        units='degrees_north',
        coord_system=coord_sys,
    )
    depth = iris.coords.DimCoord(
        [i * 100. for i in range(2, 7)],
        standard_name='depth',
        long_name='depth',
        bounds=None,
    )
    basin = iris.coords.AuxCoord(
        ['atlantic_arctic_ocean', 'indian_pacific_ocean', 'global_ocean'],
        standard_name='region',
        long_name='atlantic_arctic_ocean',
        bounds=None,
    )
    coords_spec = [(depth, 0), (lats, 1), (lons, 2)]

    cube = iris.cube.Cube(data,
                          dim_coords_and_dims=coords_spec,
                          standard_name=std_name)
    cube.add_aux_coord(basin, data_dims=[3, ])

    return cube


def test_amoc_derivation():
    """Test the actual computation for amoc."""
    msftmyz_name = 'ocean_meridional_overturning_mass_streamfunction'
    msftyz_name = 'ocean_y_overturning_mass_streamfunction'

    derived_var = amoc.DerivedVariable()

    cmip5_cube = build_ocean_cube(msftmyz_name)
    cmip6_cube = build_ocean_cube(msftyz_name)

    cmip6_cube.coord("latitude").standard_name = "grid_latitude"

    cmip5_cubes = iris.cube.CubeList([cmip5_cube])
    cmip6_cubes = iris.cube.CubeList([cmip6_cube])

    der = derived_var.calculate(cmip6_cubes)
    assert der.coord("depth").points[0] == 550.0
    assert der.coord("grid_latitude").points[0] == 26

    der = derived_var.calculate(cmip5_cubes)
    assert der.coord("depth").points[0] == 550.0
    assert der.coord("latitude").points[0] == 26
