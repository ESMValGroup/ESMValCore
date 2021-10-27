"""Tests for the fixes of CORDEX."""
import iris
import iris.coord_systems
from iris.coords import AuxCoord, DimCoord

from esmvalcore.cmor._fixes.cordex.project import AllVars
from esmvalcore.cmor.fix import Fix


def test_get_allvars_fix():
    fix = Fix.get_fixes('CORDEX', 'any_dataset', 'mip', 'tas')
    assert fix == [AllVars(None)]


def test_allvars():
    cube = iris.cube.Cube(
        [[1.0, 1.0], [1.0, 1.0]],
        var_name='co2',
        standard_name='mole_fraction_of_carbon_dioxide_in_air',
        units='mol mol-1',
    )
    # Rotated grids
    grid_latitude = DimCoord(, units=degress)
    grid_longitude = DimCoord(, units=degress)
    latitude = AuxCoord()
    longitude = AuxCoord()
    coord_sys_rotated = iris.coord_systems.RotatedGeogCS(39.25, -162.0)
    grid_lat_11 = iris.coords.DimCoord(np.linspace(-23.375, 21.835, 412),
                                    var_name='rlat',
                                    standard_name='grid_latitude',
                                    long_name='latitude in rotated-pole grid',
                                    units='degrees',
                                    coord_system=coord_sys_rotated)
    grid_lon_11 = iris.coords.DimCoord(np.linspace(-28.375, 18.155, 424),
                                    var_name='rlon',
                                    standard_name='grid_longitude',
                                    long_name='longitude in rotated-pole grid',
                                    units='degrees',
                                    coord_system=coord_sys_rotated)


def test_allvars_lambert_conformal():
    cube = iris.cube.Cube(
        [[1.0, 1.0], [1.0, 1.0]],
        var_name='co2',
        standard_name='mole_fraction_of_carbon_dioxide_in_air',
        units='mol mol-1',
    )
    # Rotated grids
    projection_x_coordinate = DimCoord([-1000, 1000],units=m, coord_system=iris.coord_systems.LambertConformal)


    LambertConformal(central_lat=48.0, central_lon=9.75, false_easting=-6000.0, false_northing=-6000.0, secant_latitudes=(30.0, 65.0), ellipsoid=GeogCS(semi_major_axis=6371229.0, semi_minor_axis=-inf))

    grid_longitude = DimCoord()
