"""Tests for the fixes of ORAS5."""

import datetime

import dask.array as da
import numpy as np
import pytest
from cf_units import Unit
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube, CubeList

from esmvalcore.cmor._fixes.fix import Fix, GenericFix
from esmvalcore.cmor._fixes.oras5.oras5 import (
    AllVars)
from esmvalcore.cmor.fix import fix_metadata
from esmvalcore.cmor.table import CMOR_TABLES, get_var_info
from esmvalcore.preprocessor import cmor_check_metadata

COMMENT = (
    "Contains modified Copernicus Climate Change Service Information "
    f"{datetime.datetime.now().year}"
)

def test_get_frequency_monthly():
    """Test cubes with monthly frequency."""
    time = DimCoord(
        [0, 31, 59],
        standard_name="time",
        units=Unit("hours since 1900-01-01"),
    )
    cube = Cube(
        [1, 6, 3],
        var_name="random_var",
        dim_coords_and_dims=[(time, 0)],
    )
    cube.coord("time").convert_units("days since 1850-1-1 00:00:00.0")


def _oras5_latitude():
    return DimCoord(
        np.array([90.0, 0.0, -90.0]),
        standard_name="grid_latitude",
        long_name="grid_latitude",
        var_name="grid_latitude",
        units=Unit("degrees"),
    )


def _oras5_longitude():
    return DimCoord(
        np.array([0, 180, 359.75]),
        standard_name="grid_longitude",
        long_name="grid_longitude",
        var_name="grid_longitude",
        units=Unit("degrees"),
        circular=True,
    )


def _oras5_time(frequency):
    if frequency == "invariant":
        timestamps = [788928]  # hours since 1900 at 1 january 1990
    elif frequency == "monthly":
        timestamps = [788928, 789672, 790344]
    else:
        raise NotImplementedError(f"Invalid frequency {frequency}")
    return DimCoord(
        np.array(timestamps, dtype="int32"),
        standard_name="time",
        long_name="time",
        var_name="time",
        units=Unit("hours since 1900-01-0100:00:00.0", calendar="gregorian"),
    )


def _oras5_depth():
    values = np.array(
        [
            0.5,
            6000,
        ]
    )
    return DimCoord(
        values,
        long_name="Vertical T levels",
        units=Unit("m"),
        var_name="deptht",
        attributes={"positive": "down"},
    )


def _oras5_data(frequency):
    if frequency == "invariant":
        return np.arange(9).reshape(1, 3, 3)
    return np.arange(27).reshape(3, 3, 3)


def _cmor_latitude():
    return DimCoord(
        np.array([-90.0, 0.0, 90.0]),
        standard_name="latitude",
        long_name="Latitude",
        var_name="lat",
        units=Unit("degrees_north"),
        bounds=np.array([[-90.0, -45.0], [-45.0, 45.0], [45.0, 90.0]]),
    )


def _cmor_longitude():
    return DimCoord(
        np.array([0, 180, 359.75]),
        standard_name="longitude",
        long_name="Longitude",
        var_name="lon",
        units=Unit("degrees_east"),
        bounds=np.array([[-0.125, 90.0], [90.0, 269.875], [269.875, 359.875]]),
        circular=True,
    )


def _cmor_time(mip, bounds=None, shifted=False):
    """Provide expected time coordinate after fixes."""
    if "mon" in mip:
        timestamps = np.array([51149.5, 51179.0, 51208.5])
        if bounds is not None:
            bounds = np.array(
                [[51134.0, 51165.0], [51165.0, 51193.0], [51193.0, 51224.0]]
            )
    else:
        raise NotImplementedError()
    
    return DimCoord(
        np.array(timestamps, dtype=float),
        standard_name="time",
        long_name="time",
        var_name="time",
        units=Unit("days since 1850-1-1 00:00:00", calendar="gregorian"),
        bounds=bounds,
    )


def _cmor_depth():
    values = np.array(
        [
            0.5,
            6000.0,
        ]
    )
    return DimCoord(
        values,
        long_name="ocean depth coordinate",
        standard_name="depth",
        units=Unit("m"),
        var_name="lev",
        attributes={"positive": "down"},
    )


def _cmor_data(mip):
    if mip == "fx":
        return np.arange(9).reshape(3, 3)[::-1, :]
    return np.arange(27).reshape(3, 3, 3)[:, ::-1, :]


def oras5_2d(frequency):    
    if frequency == "monthly":
        time = DimCoord(
            [-31, 0, 31], standard_name="time", units="days since 1850-01-01"
        )
    else:
        time = _oras5_time(frequency)
    cube = Cube(
        _oras5_data("monthly"),
        long_name=None,
        var_name=None,
        units="unknown",
        dim_coords_and_dims=[
            (time, 0),
            (_oras5_latitude(), 1),
            (_oras5_longitude(), 2),
        ],
    )
    return CubeList([cube])


def oras5_3d(frequency):
    cube = Cube(
        np.ones((3, 2, 3, 3)),
        long_name=None,
        var_name=None,
        units="unknown",
        dim_coords_and_dims=[
            (_oras5_time(frequency), 0),
            (_oras5_depth(), 1),
            (_oras5_latitude(), 2),
            (_oras5_longitude(), 3),
        ],
    )
    return CubeList([cube])

def cmor_2d(mip, short_name):
    cmor_table = CMOR_TABLES["ORAS5"]
    vardef = cmor_table.get_variable(mip, short_name)
    if "mon" in mip:
        time = DimCoord(
            [-15.5, 15.5, 45.0],
            bounds=[[-31.0, 0.0], [0.0, 31.0], [31.0, 59.0]],
            standard_name="time",
            long_name="time",
            var_name="time",
            units="days since 1850-01-01",
        )
    else:
        time = _cmor_time(mip, bounds=True)
    cube = Cube(
        _cmor_data(mip).astype("float32"),
        long_name=vardef.long_name,
        var_name=vardef.short_name,
        standard_name=vardef.standard_name,
        units=Unit(vardef.units),
        dim_coords_and_dims=[
            (time, 0),
            (_cmor_latitude(), 1),
            (_cmor_longitude(), 2),
        ],
        attributes={"comment": COMMENT},
    )
    return CubeList([cube])


def cmor_3d(mip, short_name):
    cmor_table = CMOR_TABLES["ORAS5"]
    vardef = cmor_table.get_variable(mip, short_name)
    cube = Cube(
        np.ones((3, 2, 3, 3)),
        long_name=vardef.long_name,
        var_name=vardef.short_name,
        standard_name=vardef.standard_name,
        units=Unit(vardef.units),
        dim_coords_and_dims=[
            (_cmor_time(mip, bounds=True), 0),
            (_cmor_depth(), 1),
            (_cmor_latitude(), 2),
            (_cmor_longitude(), 3),
        ],
        attributes={"comment": COMMENT},
    )
    return CubeList([cube])


# VARIABLES = [
#     pytest.param(a, b, c, d, id=c + "_" + d)
#     for (a, b, c, d) in [
#         (oras5_3d("monthly"), cmor_3d("Omon", "so"), "vosaline", "Omon"),
#         (oras5_3d("monthly"), cmor_3d("Omon", "thetao"), "votemper", "Omon"),
#         (oras5_3d("monthly"), cmor_3d("Omon", "uo"), "vozocrte", "Omon"),
#         (oras5_3d("monthly"), cmor_3d("Omon", "vo"), "vozocrte", "Omon"),
#         (oras5_2d("monthly"), cmor_2d("Omon", "mlotst"), "somxl010", "Omon"),
#         (oras5_2d("monthly"), cmor_2d("Omon", "tos"), "sosstsst", "Omon"),
#         (oras5_2d("monthly"), cmor_2d("Omon", "sos"), "sosaline", "Omon"),
#         (oras5_2d("monthly"), cmor_2d("Omon", "zos"), "sossheig", "Omon"),
#     ]
# ]



def tos_oras5_monthly():
    time = _oras5_time("monthly")
    cube = Cube(
        _oras5_data("monthly"),
        long_name="Sea Surface Temperature",
        var_name="sosstsst",
        units="degC",
        dim_coords_and_dims=[
            (time, 0),
            (_oras5_latitude(), 1),
            (_oras5_longitude(), 2),
        ],
    )
    return CubeList([cube])


def tos_cmor_omon():
    cmor_table = CMOR_TABLES["ORAS5"]
    vardef = cmor_table.get_variable("Omon", "tas")
    time = _cmor_time("Omon", bounds=True)
    data = _cmor_data("Omon")
    cube = Cube(
        data.astype("float32"),
        long_name=vardef.long_name,
        var_name=vardef.short_name,
        standard_name=vardef.standard_name,
        units=Unit(vardef.units),
        dim_coords_and_dims=[
            (time, 0),
            (_cmor_latitude(), 1),
            (_cmor_longitude(), 2),
        ],
        attributes={"comment": COMMENT},
    )
    return CubeList([cube])

VARIABLES = [
    pytest.param(a, b, c, d, id=c + "_" + d)
    for (a, b, c, d) in [
        # (oras5_3d("monthly"), cmor_3d("Omon", "so"), "so", "Omon"),
        # (oras5_3d("monthly"), cmor_3d("Omon", "thetao"), "thetao", "Omon"),
        # (oras5_3d("monthly"), cmor_3d("Omon", "uo"), "uo", "Omon"),
        # (oras5_3d("monthly"), cmor_3d("Omon", "vo"), "vo", "Omon"),
        # (oras5_2d("monthly"), cmor_2d("Omon", "mlotst"), "mlotst", "Omon"),
        # (oras5_2d("monthly"), cmor_2d("Omon", "tos"), "tos", "Omon"),
        # (oras5_2d("monthly"), cmor_2d("Omon", "sos"), "sosaline", "Omon"),
        (tos_oras5_monthly(), tos_cmor_omon(), "sosstsst", "Omon"),
        # (oras5_2d("monthly"), cmor_2d("Omon", "zos"), "zos", "Omon"),
    ]
]

@pytest.mark.parametrize("oras5_cubes, cmor_cubes, var, mip", VARIABLES)
def test_cmorization(oras5_cubes, cmor_cubes, var, mip):
    """Verify that cmorization results in the expected target cube."""
    fixed_cubes = fix_metadata(oras5_cubes, var, "ORAS5", "oras5", mip)

    assert len(fixed_cubes) == 1
    fixed_cube = fixed_cubes[0]
    cmor_cube = cmor_cubes[0]

    # Test that CMOR checks are passing
    fixed_cubes = cmor_check_metadata(fixed_cube, "ORAS5", mip, var)

    if fixed_cube.coords("time"):
        for cube in [fixed_cube, cmor_cube]:
            coord = cube.coord("time")
            coord.points = np.round(coord.points, decimals=7)
            if coord.bounds is not None:
                coord.bounds = np.round(coord.bounds, decimals=7)
    print("Test results for variable/MIP: ", var, mip)
    print("cmor_cube:", cmor_cube)
    print("fixed_cube:", fixed_cube)
    print("cmor_cube data:", cmor_cube.data)
    print("fixed_cube data:", fixed_cube.data)
    print("cmor_cube coords:")
    for coord in cmor_cube.coords():
        print(coord)
    print("\n")
    print("fixed_cube coords:")
    for coord in fixed_cube.coords():
        print(coord)
    assert fixed_cube == cmor_cube


@pytest.fixture
def unstructured_grid_cubes():
    """Sample cubes with unstructured grid."""
    time = DimCoord(
        [0.0, 31.0], standard_name="time", units="days since 1950-01-01"
    )
    lat = AuxCoord(
        [1.0, 1.0, -1.0, -1.0], standard_name="latitude", units="degrees_north"
    )
    lon = AuxCoord(
        [179.0, 180.0, 180.0, 179.0],
        standard_name="longitude",
        units="degrees_east",
    )
    cube = Cube(
        da.from_array([[0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 0.0]]),
        standard_name="sea_surface_salinity",
        units="0.001",
        dim_coords_and_dims=[(time, 0)],
        aux_coords_and_dims=[(lat, 1), (lon, 1)],
    )
    return CubeList([cube])


def test_unstructured_grid(unstructured_grid_cubes):
    """Test processing unstructured data."""
    fixed_cubes = fix_metadata(
        unstructured_grid_cubes,
        "sosaline",
        "ORAS5",
        "oras5",
        "Omon",
    )

    assert len(fixed_cubes) == 1
    fixed_cube = fixed_cubes[0]

    assert fixed_cube.shape == (2, 4)

    assert fixed_cube.coords("time", dim_coords=True)
    assert fixed_cube.coord_dims("time") == (0,)

    assert fixed_cube.coords("latitude", dim_coords=False)
    assert fixed_cube.coord_dims("latitude") == (1,)
    lat = fixed_cube.coord("latitude")
    np.testing.assert_allclose(lat.points, [1, 1, -1, -1])
    assert lat.bounds is None

    assert fixed_cube.coords("longitude", dim_coords=False)
    assert fixed_cube.coord_dims("longitude") == (1,)
    lon = fixed_cube.coord("longitude")
    np.testing.assert_allclose(lon.points, [179, 180, 180, 179])
    assert lon.bounds is None


