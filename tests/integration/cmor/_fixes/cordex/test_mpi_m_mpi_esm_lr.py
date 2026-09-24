"""Tests for the fixes of driver MPI-M-MPI-ESM-LR."""

import cordex as cx
import dask.array as da
import iris
import iris.coords
import iris.cube
import numpy as np
import pytest

from esmvalcore.cmor._fixes.cordex.cordex_fixes import CLMcomCCLM4817
from esmvalcore.cmor._fixes.cordex.mpi_m_mpi_esm_lr import (
    cosmo_crclim_v1_1,
)
from esmvalcore.cmor.fix import Fix


@pytest.mark.parametrize("short_name", ["pr", "tas"])
def test_get_regcm4_6_fix(short_name):
    fix = Fix.get_fixes(
        "CORDEX",
        "REGCM4-6",
        "Amon",
        short_name,
        extra_facets={"driver": "MPI-M-MPI-ESM-LR"},
    )
    assert isinstance(fix[0], Fix)


def test_get_racmo22e_fix():
    fix = Fix.get_fixes(
        "CORDEX",
        "RACMO22E",
        "Amon",
        "pr",
        extra_facets={"driver": "MPI-M-MPI-ESM-LR"},
    )
    assert isinstance(fix[0], Fix)


@pytest.mark.parametrize("short_name", ["pr", "tas"])
def test_get_hadrem3ga705_fix(short_name):
    fix = Fix.get_fixes(
        "CORDEX",
        "HadREM3-GA7-05",
        "Amon",
        short_name,
        extra_facets={"driver": "MPI-M-MPI-ESM-LR"},
    )
    assert isinstance(fix[0], Fix)


def test_get_cclm4_8_17fix() -> None:
    fixes = Fix.get_fixes(
        "CORDEX",
        "CCLM4-8-17",
        "Amon",
        "ts",
        extra_facets={"driver": "MPI-M-MPI-ESM-LR"},
    )
    assert any(isinstance(fix, CLMcomCCLM4817) for fix in fixes)


def test_get_cosmo_crclim_v1_1_fix() -> None:
    fixes = Fix.get_fixes(
        "CORDEX",
        "COSMO-crCLIM-v1-1",
        "day",
        "snw",
        extra_facets={"driver": "MPI-M-MPI-ESM-LR"},
    )
    assert any(isinstance(fix, cosmo_crclim_v1_1.Snw) for fix in fixes)


def test_wrf381p_rlut_standard_grid() -> None:

    standard_grid = cx.domain("EUR-11")
    cube = iris.cube.Cube(
        da.empty((1, 412, 424), dtype="float32"),
        var_name="rlut",
        standard_name="toa_outgoing_longwave_flux",
        long_name="TOA Outgoing Longwave Radiation",
        units="W m-2",
        dim_coords_and_dims=[
            (
                iris.coords.DimCoord(
                    [0],
                    bounds=[[-0.5, 0.5]],
                    var_name="time",
                    standard_name="time",
                    units="days since 2000-01-01",
                ),
                0,
            ),
        ],
        aux_coords_and_dims=[
            (
                iris.coords.AuxCoord(
                    np.zeros(412),
                    var_name="y",
                    standard_name="projection_y_coordinate",
                    units="m",
                ),
                (1,),
            ),
            (
                iris.coords.AuxCoord(
                    np.zeros(424),
                    var_name="x",
                    standard_name="projection_x_coordinate",
                    units="m",
                ),
                (2,),
            ),
            (
                iris.coords.AuxCoord(
                    standard_grid.lat.data + 1e-7,
                    var_name="lat",
                    standard_name="latitude",
                    units="degrees_north",
                ),
                (1, 2),
            ),
            (
                iris.coords.AuxCoord(
                    standard_grid.lon.data + 1e-7,
                    var_name="lon",
                    standard_name="longitude",
                    units="degrees_east",
                ),
                (1, 2),
            ),
        ],
    )
    fixes = Fix.get_fixes(
        "CORDEX",
        "WRF381P",
        "day",
        "rlut",
        extra_facets={
            "driver": "MPI-M-MPI-ESM-LR",
            "domain": "EUR-11",
        },
    )
    cubes = [cube]
    for fix in fixes:
        cubes = fix.fix_metadata(cubes)
    assert len(cubes) == 1
    result = cubes[0]
    for standard_name, var_name in [
        ("grid_latitude", "rlat"),
        ("grid_longitude", "rlon"),
        ("latitude", "lat"),
        ("longitude", "lon"),
    ]:
        print("Checking coordinate:", standard_name)
        coord = result.coord(standard_name)
        var = standard_grid[var_name]
        assert coord.units == var.attrs["units"]
        expected_points = (
            var.data % 360 if standard_name == "longitude" else var.data
        )
        np.testing.assert_almost_equal(coord.points, expected_points)
        assert coord.has_bounds()
