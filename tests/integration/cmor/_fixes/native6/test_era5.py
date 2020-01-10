"""Tests for the fixes of ERA5."""
import iris
import numpy as np
import pytest
from cf_units import Unit

from esmvalcore.cmor._fixes.native6.era5 import Evspsbl, Pr, AllVars
from esmvalcore.cmor.fix import Fix
from esmvalcore.cmor.fix import fix_file

def test_get_evspsbl_fix():
    """Test whether the right fixes are gathered for a single variable."""
    fix = Fix.get_fixes('native6', 'ERA5', 'E1hr', 'evspsbl')
    assert fix == [AllVars(None), Evspsbl(None)]


@pytest.fixture
def pr_src_cube():
    """Make dummy cube that looks like the ERA5 data."""
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
    pr = iris.cube.Cube(np.zeros((3, 3, 3)),
                        long_name='Total precipitation',
                        var_name='tp',
                        units=Unit('m'),
                        dim_coords_and_dims=[(time, 0), (latitude, 1), (longitude, 2)],
                        attributes= {'Conventions': 'CF-1.6',
                                     'History': 'uninteresting info'})
    return iris.cube.CubeList([pr])


@pytest.fixture
def pr_target_cube():
    """Make dummy cube that looks like the CMORized data."""
    # TODO: Verify that all cmor standards are accounted for in this
    # cube. I addressed: lat/lon var_name; latitude direction;
    # coordinate bounds; variable metadata from cmor table; ...
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
    pr = iris.cube.Cube(np.zeros((3, 3, 3)),
                        long_name='Precipitation',
                        var_name='pr',
                        standard_name='precipitation_flux',
                        units=Unit('kg m-2 s-1'),
                        dim_coords_and_dims=[(time, 0), (latitude, 1), (longitude, 2)],
                        attributes= {'Conventions': 'CF-1.5',
                                     'History': 'uninteresting info',
                                     'More': 'uninteresting stuff'})
    return iris.cube.CubeList([pr])

def test_cmorization(pr_src_cubes,  pr_target_cubes):
    """Verify that cmorization results in the expected target cube."""
    fix = Pr()
    out_cubes = fix.fix_metadata(pr_src_cubes)
    assert out_cubes == target_cubes

# TODO:
# - Make the dummy cube function more generic, such that it can be reused for multiple variables
# - Make a pytest parameterize function to test for multiple variables, make sure to include all variables that have a specific class and 1 general variable.
# - Include tests for monthly data and 3D variables (1 variable is enough)
