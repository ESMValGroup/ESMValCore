"""Tests for the fixes of IITM-ESM."""
import iris
import numpy as np
import pytest
from cf_units import Unit

from esmvalcore.cmor._fixes.cmip6.iitm_esm import AllVars, Tos
from esmvalcore.cmor._fixes.common import OceanFixGrid
from esmvalcore.cmor._fixes.fix import GenericFix
from esmvalcore.cmor.fix import Fix


def test_get_tos_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes('CMIP6',
                        'IITM-ESM',
                        'Omon',
                        'tos',
                        extra_facets={"frequency": "mon"})
    assert fix == [Tos(None), AllVars(None), GenericFix(None)]


def test_tos_fix():
    """Test fix for ``tos``."""
    assert Tos is OceanFixGrid


@pytest.fixture
def cubes():
    correct_time_coord = iris.coords.DimCoord(
        [15.5, 45, 74.5],
        bounds=[[0., 31.], [31., 59.], [59., 90.]],
        var_name='time',
        standard_name='time',
        units=Unit('days since 0001-01-01 00:00:00', calendar='365_day'))
    wrong_time_coord = iris.coords.DimCoord(
        [15.5, 45, 74.5],
        bounds=[[5.5, 25.5], [35., 55.], [64.5, 84.5]],
        var_name='time',
        standard_name='time',
        units=Unit('days since 0001-01-01 00:00:00', calendar='365_day'))

    correct_lat_coord = iris.coords.DimCoord([0.0, 1.0],
                                             bounds=[[-0.5, 0.5], [0.5, 1.5]],
                                             var_name='lat',
                                             standard_name='latitude',
                                             units='degrees')

    correct_lon_coord = iris.coords.DimCoord([0.0, 1.0],
                                             bounds=[[-0.5, 0.5], [0.5, 1.5]],
                                             var_name='lon',
                                             standard_name='longitude',
                                             units='degrees')

    correct_cube = iris.cube.Cube(10 * np.ones((3, 2, 2)),
                                  var_name='tos',
                                  dim_coords_and_dims=[(correct_time_coord, 0),
                                                       (correct_lat_coord, 1),
                                                       (correct_lon_coord, 2)],
                                  attributes={'table_id': 'Omon'},
                                  units=Unit('degC'))

    wrong_cube = iris.cube.Cube(10 * np.ones((3, 2, 2)),
                                var_name='tos',
                                dim_coords_and_dims=[(wrong_time_coord, 0),
                                                     (correct_lat_coord, 1),
                                                     (correct_lon_coord, 2)],
                                attributes={'table_id': 'Omon'},
                                units=Unit('degC'))

    return iris.cube.CubeList([correct_cube, wrong_cube])


def test_allvars_fix_metadata(monkeypatch, cubes, caplog):
    fix = AllVars(None)
    monkeypatch.setitem(fix.extra_facets, 'frequency', 'mon')
    monkeypatch.setitem(fix.extra_facets, 'dataset', 'IITM-ESM')
    out_cubes = fix.fix_metadata(cubes)
    assert cubes is out_cubes
    for cube in out_cubes:
        time = cube.coord('time')
        assert all(time.bounds[1:, 0] == time.bounds[:-1, 1])
    msg = ("Using 'area_weighted' regridder scheme in Omon variables "
           "for dataset IITM-ESM causes discontinuities in the longitude "
           "coordinate.")
    assert msg in caplog.text
