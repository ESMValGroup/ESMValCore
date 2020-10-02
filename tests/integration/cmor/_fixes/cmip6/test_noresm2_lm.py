"""Tests for the fixes of GFDL-ESM4."""
import os

import iris
import numpy as np
import pytest
from netCDF4 import Dataset

from esmvalcore.cmor._fixes.cmip6.noresm2_lm import Siconc
from esmvalcore.cmor.fix import Fix
from esmvalcore.cmor.table import get_var_info


@pytest.fixture
def siconc_file(tmp_path):
    """Create netcdf file."""
    nc_path = os.path.join(tmp_path, 'noresm2_lm_siconc.nc')
    with Dataset(nc_path, mode='w') as dataset:
        dataset.createDimension('time', size=1)
        dataset.createDimension('lat', size=1)
        dataset.createDimension('lon', size=1)

        # Dimensional variables
        dataset.createVariable('time', np.float64, dimensions=('time',))
        dataset.createVariable('lat', np.float64, dimensions=('lat',))
        dataset.createVariable('lon', np.float64, dimensions=('lon',))
        dataset.variables['time'][:] = [0.2]
        dataset.variables['time'].standard_name = 'time'
        dataset.variables['time'].units = 'days since 1850-01-01'
        dataset.variables['lat'][:] = [30.0]
        dataset.variables['lat'].standard_name = 'latitude'
        dataset.variables['lat'].units = 'degrees_north'
        dataset.variables['lon'][:] = [30.0]
        dataset.variables['lon'].standard_name = 'longitude'
        dataset.variables['lon'].units = 'degrees_east'
        dataset.createVariable('siconc', np.float64,
                               dimensions=('time', 'lat', 'lon'))
        dataset.variables['siconc'][:] = 22.
        dataset.variables['siconc'].standard_name = 'sea_ice_area_fraction'
        dataset.variables['siconc'].units = '%'

    return nc_path


def test_get_siconc_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6', 'NorESM2-LM', 'SImon', 'siconc')
    assert fix == [Siconc(None)]


def test_siconc_fix_metadata(siconc_file):
    """Test ``fix_metadata``."""
    print(siconc_file)
    cubes = iris.load(siconc_file)
    for cube in cubes:
        cube.coord("latitude").bounds = [28.9956255, 32.3445677]
        cube.coord("longitude").bounds = [28.9956255, 32.3445677]

    # Raw cubes
    assert len(cubes) == 1
    siconc_cube = cubes[0]
    assert siconc_cube.var_name == "siconc"

    # Extract siconc cube
    siconc_cube = cubes.extract_strict('sea_ice_area_fraction')
    assert not siconc_cube.coords('typesi')

    # Apply fix
    vardef = get_var_info('CMIP6', 'SImon', 'siconc')
    fix = Siconc(vardef)
    fixed_cubes = fix.fix_metadata(cubes)
    assert len(fixed_cubes) == 1
    fixed_siconc_cube = fixed_cubes.extract_strict(
        'sea_ice_area_fraction')
    fixed_lon = fixed_siconc_cube.coord('longitude')
    fixed_lat = fixed_siconc_cube.coord('latitude')
    assert fixed_lon.bounds is not None
    assert fixed_lat.bounds is not None
    np.testing.assert_equal(fixed_lon.bounds, [[28.9956, 32.3446]])
    np.testing.assert_equal(fixed_lat.bounds, [[28.9956, 32.3446]])
