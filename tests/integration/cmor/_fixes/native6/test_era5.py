"""Tests for the fixes of ERA5."""
import iris
import numpy as np
import pytest
from cf_units import Unit

from esmvalcore.cmor._fixes.native6.era5 import AllVars, Evspsbl
from esmvalcore.cmor.fix import Fix, fix_metadata
from esmvalcore.cmor.table import CMOR_TABLES


def test_get_evspsbl_fix():
    """Test whether the right fixes are gathered for a single variable."""
    fix = Fix.get_fixes('native6', 'ERA5', 'E1hr', 'evspsbl')
    assert fix == [Evspsbl(None), AllVars(None)]


def make_src_cubes(long_name, var_name, units, ndims=3, mip='E1hr'):
    """Make dummy cube that looks like the ERA5 data."""
    # TODO: Make 2d and 4d dimensions possible
    latitude = iris.coords.DimCoord(np.array([90., 0., -90.]),
                                    standard_name='latitude',
                                    long_name='latitude',
                                    var_name='latitude',
                                    units=Unit('degrees'))
    longitude = iris.coords.DimCoord(np.array([0, 180, 359.75]),
                                     standard_name='longitude',
                                     long_name='longitude',
                                     var_name='longitude',
                                     units=Unit('degrees'),
                                     circular=True)
    time = iris.coords.DimCoord(np.arange(788928, 788931, dtype='int32'),
                                standard_name='time',
                                long_name='time',
                                var_name='time',
                                units=Unit('hours since 1900-01-01 00:00:00.0', calendar='gregorian'))

    cube = iris.cube.Cube(np.zeros((3, 3, 3)),
                          long_name=long_name,
                          var_name=var_name,
                          units=Unit(units),
                          dim_coords_and_dims=[(time, 0), (latitude, 1), (longitude, 2)],
                          attributes={'Conventions': 'CF-1.6',
                                      'History': 'uninteresting info'})
    return iris.cube.CubeList([cube])


def make_target_cubes(long_name, var_name, standard_name, units, ndims=3, mip='E1hr'):
    """Make dummy cube that looks like the CMORized data."""
    # TODO: Verify that all cmor standards are accounted for in this
    # cube. I addressed: lat/lon var_name; latitude direction;
    # coordinate bounds; variable metadata from cmor table; ...
    # TODO: Make 2d and 4d dimensions possible
    latitude = iris.coords.DimCoord(np.array([-90., 0., 90.]),
                                    standard_name='latitude',
                                    long_name='latitude',
                                    var_name='lat',
                                    units=Unit('degrees'),
                                    bounds=-np.array([[-89.875, -90],
                                                      [-0.125, 0.125],
                                                      [90, 89.875]]))
    longitude = iris.coords.DimCoord(np.array([0, 180, 359.75]),
                                     standard_name='longitude',
                                     long_name='longitude',
                                     var_name='lon',
                                     units=Unit('degrees'),
                                     bounds=np.array([[-0.125, 0.125],
                                                      [179.875, 180.125],
                                                      [359.625, 359.875]]),
                                     circular=True)
    time = iris.coords.DimCoord(np.arange(788928, 788931, dtype='int32'),
                                standard_name='time',
                                long_name='time',
                                var_name='time',
                                units=Unit('hours since 1900-01-01 00:00:00.0', calendar='gregorian'),
                                bounds=np.array([[788927.5, 788928.5],
                                                 [788928.5, 788929.5],
                                                 [788929.5, 788930.5]]))
    cube = iris.cube.Cube(np.zeros((3, 3, 3)),
                          long_name=long_name,
                          var_name=var_name,
                          standard_name=standard_name,
                          units=Unit(units),
                          dim_coords_and_dims=[(time, 0), (latitude, 1),   (longitude, 2)],
                          attributes={'Conventions': 'CF-1.5',
                                      'History': 'uninteresting info',
                                      'More': 'uninteresting stuff'})
    return [cube]


variables = [
    # short_name, mip, era5_units, ndims
    ['pr', 'E1hr', 'm', 3],
    ['evspsbl', 'E1hr', 'm', 3],
    # ['mrro', 'E1hr', 'm', 3],
    # ['prsn', 'E1hr', 'm of water equivalent', 3],
    # ['evspsblpot', 'E1hr', 'm', 3],
    # ['rss', 'E1hr', 'J m**-2', 3],
    # ['rsds', 'E1hr', 'J m**-2', 3],
    # ['rsdt', 'E1hr', 'J m**-2', 3],
    # ['rls', 'E1hr', 'W m**-2', 3], # variables with explicit fixes
    # ['uas', 'E1hr', 'm s**-1', 3], # a variable without explicit fixes
    # ['pr', 'Amon', 'm', 3], # a monthly variable
    # ['ua', 'E1hr', 'm s**-1', 4], # a 4d variable
    # ['orog', 'Fx', 'm**2 s**-2'] # ?? # a funky 2D variable
]


@pytest.mark.parametrize('variable', variables)
def test_cmorization(variable):
    """Verify that cmorization results in the expected target cube."""
    short_name, mip, era5_units, ndims = variable
    project, dataset = 'native6', 'era5'

    # Look up variable definition in CMOR table
    cmor_table = CMOR_TABLES[project]
    vardef = cmor_table.get_variable(mip, short_name)
    long_name = vardef.long_name
    var_name = short_name # vardef.cmor_name after #391?
    standard_name = vardef.standard_name
    units = vardef.units

    src_cubes = make_src_cubes('era5_long_name', 'era5_var_name', era5_units)
    target_cubes = make_target_cubes(long_name, var_name, standard_name, units)
    out_cubes = fix_metadata(src_cubes, short_name, project, dataset, mip)
    print(out_cubes[0])
    print(target_cubes[0])
    assert out_cubes[0] == target_cubes[0]

# TODO:
# The test now fails because AllVars is fixed before FixUnits.
