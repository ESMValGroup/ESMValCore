"""Tests for the fixes of GFDL-ESM4."""
import os

import iris
import numpy as np
import pytest
from cf_units import Unit
from netCDF4 import Dataset

from esmvalcore.cmor._fixes.cmip6.gfdl_esm4 import Siconc
from esmvalcore.cmor.fix import Fix
from esmvalcore.cmor.table import get_var_info


@pytest.fixture
def siconc_file(tmp_path):
    """Create netcdf file with similar issues as ``cl``."""
    nc_path = os.path.join(tmp_path, 'gfdl_esm4_siconc.nc')
    with Dataset(nc_path, mode='w') as dataset:
        dataset.createDimension('time', size=1)
        dataset.createDimension('lat', size=1)
        dataset.createDimension('lon', size=1)

        # Dimensional variables
        dataset.createVariable('time', np.float64, dimensions=('time',))
        dataset.createVariable('lat', np.float64, dimensions=('lat',))
        dataset.createVariable('lon', np.float64, dimensions=('lon',))
        dataset.variables['time'][:] = [0.0]
        dataset.variables['time'].standard_name = 'time'
        dataset.variables['time'].units = 'days since 6543-2-1'
        dataset.variables['lat'][:] = [-30.0]
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
    fix = Fix.get_fixes('CMIP6', 'GFDL-ESM4', 'SImon', 'siconc')
    assert fix == [Siconc(None)]


def test_siconc_fix_metadata(siconc_file):
    """Test ``fix_metadata`` for ``cl``."""
    cubes = iris.load(siconc_file)

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
    fixed_typesi_coord = fixed_siconc_cube.coord('area_type')
    assert fixed_typesi_coord.points is not None
    assert fixed_typesi_coord.bounds is None
    np.testing.assert_equal(fixed_typesi_coord.points,
                            ['siconc'])
    np.testing.assert_equal(fixed_typesi_coord.units,
                            Unit('1'))
