"""Tests for the fixes of ERA5."""

import datetime

import dask.array as da
import numpy as np
import pytest
from cf_units import Unit
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube, CubeList

from esmvalcore.cmor._fixes.fix import Fix, GenericFix
from esmvalcore.cmor._fixes.native6.era5 import (
    AllVars,
    Evspsbl,
    Zg,
    fix_accumulated_units,
    get_frequency,
)
from esmvalcore.cmor.fix import fix_metadata
from esmvalcore.cmor.table import CMOR_TABLES, get_var_info
from esmvalcore.preprocessor import cmor_check_metadata

COMMENT = (
    "Contains modified Copernicus Climate Change Service Information "
    f"{datetime.datetime.now().year}"
)


def test_get_evspsbl_fix():
    """Test whether the right fixes are gathered for a single variable."""
    fix = Fix.get_fixes("native6", "ERA5", "E1hr", "evspsbl")
    vardef = get_var_info("native6", "E1hr", "evspsbl")
    assert fix == [Evspsbl(vardef), AllVars(vardef), GenericFix(vardef)]


def test_get_zg_fix():
    """Test whether the right fix gets found again, for zg as well."""
    fix = Fix.get_fixes("native6", "ERA5", "Amon", "zg")
    vardef = get_var_info("native6", "E1hr", "evspsbl")
    assert fix == [Zg(vardef), AllVars(vardef), GenericFix(vardef)]


def test_get_frequency_hourly():
    """Test cubes with hourly frequency."""
    time = DimCoord(
        [0, 1, 2],
        standard_name="time",
        units=Unit("hours since 1900-01-01"),
    )
    cube = Cube(
        [1, 6, 3],
        var_name="random_var",
        dim_coords_and_dims=[(time, 0)],
    )
    assert get_frequency(cube) == "hourly"
    cube.coord("time").convert_units("days since 1850-1-1 00:00:00.0")
    assert get_frequency(cube) == "hourly"


def test_get_frequency_daily():
    """Test cubes with daily frequency."""
    time = DimCoord(
        [0, 1, 2],
        standard_name="time",
        units=Unit("days since 1900-01-01"),
    )
    cube = Cube(
        [1, 6, 3],
        var_name="random_var",
        dim_coords_and_dims=[(time, 0)],
    )
    assert get_frequency(cube) == "daily"
    cube.coord("time").convert_units("hours since 1850-1-1 00:00:00.0")
    assert get_frequency(cube) == "daily"


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
    assert get_frequency(cube) == "monthly"
    cube.coord("time").convert_units("days since 1850-1-1 00:00:00.0")
    assert get_frequency(cube) == "monthly"


def test_get_frequency_fx():
    """Test cubes with time invariant frequency."""
    cube = Cube(1.0, long_name="Cube without time coordinate")
    assert get_frequency(cube) == "fx"

    time = DimCoord(
        0,
        standard_name="time",
        units=Unit("hours since 1900-01-01"),
    )
    cube = Cube(
        [1],
        var_name="cube_with_length_1_time_coord",
        long_name="Geopotential",
        dim_coords_and_dims=[(time, 0)],
    )
    assert get_frequency(cube) == "fx"

    cube.long_name = (
        "Percentage of the Grid Cell Occupied by Land (Including Lakes)"
    )
    assert get_frequency(cube) == "fx"

    cube.long_name = "Not geopotential"
    with pytest.raises(ValueError):
        get_frequency(cube)


def test_fix_accumulated_units_fail():
    """Test `fix_accumulated_units`."""
    time = DimCoord(
        [0, 1, 2],
        standard_name="time",
        units=Unit("days since 1900-01-01"),
    )
    cube = Cube(
        [1, 6, 3],
        var_name="random_var",
        dim_coords_and_dims=[(time, 0)],
    )
    with pytest.raises(NotImplementedError):
        fix_accumulated_units(cube)


def _era5_latitude():
    return DimCoord(
        np.array([90.0, 0.0, -90.0]),
        standard_name="latitude",
        long_name="latitude",
        var_name="latitude",
        units=Unit("degrees"),
    )


def _era5_longitude():
    return DimCoord(
        np.array([0, 180, 359.75]),
        standard_name="longitude",
        long_name="longitude",
        var_name="longitude",
        units=Unit("degrees"),
        circular=True,
    )


def _era5_time(frequency):
    if frequency == "invariant":
        timestamps = [788928]  # hours since 1900 at 1 january 1990
    elif frequency == "daily":
        timestamps = [788940, 788964, 788988]
    elif frequency == "hourly":
        timestamps = [788928, 788929, 788930]
    elif frequency == "monthly":
        timestamps = [788928, 789672, 790344]
    else:
        msg = f"Invalid frequency {frequency}"
        raise NotImplementedError(msg)
    return DimCoord(
        np.array(timestamps, dtype="int32"),
        standard_name="time",
        long_name="time",
        var_name="time",
        units=Unit("hours since 1900-01-0100:00:00.0", calendar="gregorian"),
    )


def _era5_plev():
    values = np.array(
        [
            1,
            1000,
        ],
    )
    return DimCoord(
        values,
        long_name="pressure",
        units=Unit("millibars"),
        var_name="level",
        attributes={"positive": "down"},
    )


def _era5_data(frequency):
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
    if mip == "E1hr":
        offset = 51134  # days since 1850 at 1 january 1990
        timestamps = offset + np.arange(3) / 24
        if shifted:
            timestamps -= 1 / 48
        if bounds is not None:
            bounds = [[t - 1 / 48, t + 1 / 48] for t in timestamps]
    elif mip == "Eday":
        timestamps = np.array([51134.5, 51135.5, 51136.5])
        if bounds is not None:
            bounds = np.array(
                [[51134.0, 51135.0], [51135.0, 51136.0], [51136.0, 51137.0]],
            )
    elif "mon" in mip:
        timestamps = np.array([51149.5, 51179.0, 51208.5])
        if bounds is not None:
            bounds = np.array(
                [[51134.0, 51165.0], [51165.0, 51193.0], [51193.0, 51224.0]],
            )
    else:
        raise NotImplementedError

    return DimCoord(
        np.array(timestamps, dtype=float),
        standard_name="time",
        long_name="time",
        var_name="time",
        units=Unit("days since 1850-1-1 00:00:00", calendar="gregorian"),
        bounds=bounds,
    )


def _cmor_aux_height(value):
    return AuxCoord(
        value,
        long_name="height",
        standard_name="height",
        units=Unit("m"),
        var_name="height",
        attributes={"positive": "up"},
    )


def _cmor_plev():
    values = np.array(
        [
            100000.0,
            100.0,
        ],
    )
    return DimCoord(
        values,
        long_name="pressure",
        standard_name="air_pressure",
        units=Unit("Pa"),
        var_name="plev",
        attributes={"positive": "down"},
    )


def _cmor_data(mip):
    if mip == "fx":
        return np.arange(9).reshape(3, 3)[::-1, :]
    return np.arange(27).reshape(3, 3, 3)[:, ::-1, :]


def era5_2d(frequency):
    if frequency == "monthly":
        time = DimCoord(
            [-31, 0, 31],
            standard_name="time",
            units="days since 1850-01-01",
        )
    else:
        time = _era5_time(frequency)
    cube = Cube(
        _era5_data(frequency),
        long_name=None,
        var_name=None,
        units="unknown",
        dim_coords_and_dims=[
            (time, 0),
            (_era5_latitude(), 1),
            (_era5_longitude(), 2),
        ],
    )
    return CubeList([cube])


def era5_3d(frequency):
    cube = Cube(
        np.ones((3, 2, 3, 3)),
        long_name=None,
        var_name=None,
        units="unknown",
        dim_coords_and_dims=[
            (_era5_time(frequency), 0),
            (_era5_plev(), 1),
            (_era5_latitude(), 2),
            (_era5_longitude(), 3),
        ],
    )
    return CubeList([cube])


def cmor_2d(mip, short_name):
    cmor_table = CMOR_TABLES["native6"]
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
    cmor_table = CMOR_TABLES["native6"]
    vardef = cmor_table.get_variable(mip, short_name)
    cube = Cube(
        np.ones((3, 2, 3, 3)),
        long_name=vardef.long_name,
        var_name=vardef.short_name,
        standard_name=vardef.standard_name,
        units=Unit(vardef.units),
        dim_coords_and_dims=[
            (_cmor_time(mip, bounds=True), 0),
            (_cmor_plev(), 1),
            (_cmor_latitude(), 2),
            (_cmor_longitude(), 3),
        ],
        attributes={"comment": COMMENT},
    )
    return CubeList([cube])


def cl_era5_monthly():
    time = _era5_time("monthly")
    data = np.ones((3, 2, 3, 3))
    cube = Cube(
        data,
        long_name="Percentage Cloud Cover",
        var_name="cl",
        units="%",
        dim_coords_and_dims=[
            (time, 0),
            (_era5_plev(), 1),
            (_era5_latitude(), 2),
            (_era5_longitude(), 3),
        ],
    )
    return CubeList([cube])


def cl_cmor_amon():
    cmor_table = CMOR_TABLES["native6"]
    vardef = cmor_table.get_variable("Amon", "cl")
    time = _cmor_time("Amon", bounds=True)
    data = np.ones((3, 2, 3, 3))
    data = data * 100.0
    cube = Cube(
        data.astype("float32"),
        long_name=vardef.long_name,
        var_name=vardef.short_name,
        standard_name=vardef.standard_name,
        units=Unit(vardef.units),
        dim_coords_and_dims=[
            (time, 0),
            (_cmor_plev(), 1),
            (_cmor_latitude(), 2),
            (_cmor_longitude(), 3),
        ],
        attributes={"comment": COMMENT},
    )
    return CubeList([cube])


def clt_era5_hourly():
    time = _era5_time("hourly")
    cube = Cube(
        _era5_data("hourly"),
        long_name="cloud cover fraction",
        var_name="cloud_cover",
        units="unknown",
        dim_coords_and_dims=[
            (time, 0),
            (_era5_latitude(), 1),
            (_era5_longitude(), 2),
        ],
    )
    return CubeList([cube])


def clt_cmor_e1hr():
    cmor_table = CMOR_TABLES["native6"]
    vardef = cmor_table.get_variable("E1hr", "clt")
    time = _cmor_time("E1hr", bounds=True)
    data = _cmor_data("E1hr") * 100
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


def evspsbl_era5_hourly():
    time = _era5_time("hourly")
    cube = Cube(
        _era5_data("hourly") * -1.0,
        long_name="total evapotranspiration",
        var_name="e",
        units="unknown",
        dim_coords_and_dims=[
            (time, 0),
            (_era5_latitude(), 1),
            (_era5_longitude(), 2),
        ],
    )
    return CubeList([cube])


def evspsbl_cmor_e1hr():
    cmor_table = CMOR_TABLES["native6"]
    vardef = cmor_table.get_variable("E1hr", "evspsbl")
    time = _cmor_time("E1hr", shifted=True, bounds=True)
    data = _cmor_data("E1hr") * 1000 / 3600.0
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


def evspsblpot_era5_hourly():
    time = _era5_time("hourly")
    cube = Cube(
        _era5_data("hourly") * -1.0,
        long_name="potential evapotranspiration",
        var_name="epot",
        units="unknown",
        dim_coords_and_dims=[
            (time, 0),
            (_era5_latitude(), 1),
            (_era5_longitude(), 2),
        ],
    )
    return CubeList([cube])


def evspsblpot_cmor_e1hr():
    cmor_table = CMOR_TABLES["native6"]
    vardef = cmor_table.get_variable("E1hr", "evspsblpot")
    time = _cmor_time("E1hr", shifted=True, bounds=True)
    data = _cmor_data("E1hr") * 1000 / 3600.0
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


def mrro_era5_hourly():
    time = _era5_time("hourly")
    cube = Cube(
        _era5_data("hourly"),
        long_name="runoff",
        var_name="runoff",
        units="m",
        dim_coords_and_dims=[
            (time, 0),
            (_era5_latitude(), 1),
            (_era5_longitude(), 2),
        ],
    )
    return CubeList([cube])


def mrro_cmor_e1hr():
    cmor_table = CMOR_TABLES["native6"]
    vardef = cmor_table.get_variable("E1hr", "mrro")
    time = _cmor_time("E1hr", shifted=True, bounds=True)
    data = _cmor_data("E1hr") * 1000 / 3600.0
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


def o3_era5_monthly():
    cube = era5_3d("monthly")[0]
    cube = cube[:, ::-1, ::-1, :]  # test if correct order of plev and lat stay
    cube.data = cube.data.astype("float32")
    cube.data *= 47.9982 / 28.9644
    return CubeList([cube])


def orog_era5_hourly():
    time = _era5_time("invariant")
    cube = Cube(
        _era5_data("invariant"),
        long_name="geopotential height",
        var_name="zg",
        units="m**2 s**-2",
        dim_coords_and_dims=[
            (time, 0),
            (_era5_latitude(), 1),
            (_era5_longitude(), 2),
        ],
    )
    return CubeList([cube])


def orog_cmor_fx():
    cmor_table = CMOR_TABLES["native6"]
    vardef = cmor_table.get_variable("fx", "orog")
    data = _cmor_data("fx") / 9.80665
    cube = Cube(
        data.astype("float32"),
        long_name=vardef.long_name,
        var_name=vardef.short_name,
        standard_name=vardef.standard_name,
        units=Unit(vardef.units),
        dim_coords_and_dims=[(_cmor_latitude(), 0), (_cmor_longitude(), 1)],
        attributes={"comment": COMMENT},
    )
    return CubeList([cube])


def pr_era5_monthly():
    time = _era5_time("monthly")
    cube = Cube(
        _era5_data("monthly"),
        long_name="total_precipitation",
        var_name="tp",
        units="m",
        dim_coords_and_dims=[
            (time, 0),
            (_era5_latitude(), 1),
            (_era5_longitude(), 2),
        ],
    )
    return CubeList([cube])


def pr_cmor_amon():
    cmor_table = CMOR_TABLES["native6"]
    vardef = cmor_table.get_variable("Amon", "pr")
    time = _cmor_time("Amon", bounds=True)
    data = _cmor_data("Amon") * 1000.0 / 3600.0 / 24.0
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


def pr_era5_hourly():
    time = _era5_time("hourly")
    cube = Cube(
        _era5_data("hourly"),
        long_name="total_precipitation",
        var_name="tp",
        units="m",
        dim_coords_and_dims=[
            (time, 0),
            (_era5_latitude(), 1),
            (_era5_longitude(), 2),
        ],
    )
    return CubeList([cube])


def pr_cmor_e1hr():
    cmor_table = CMOR_TABLES["native6"]
    vardef = cmor_table.get_variable("E1hr", "pr")
    time = _cmor_time("E1hr", bounds=True, shifted=True)
    data = _cmor_data("E1hr") * 1000.0 / 3600.0
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


def prsn_era5_hourly():
    time = _era5_time("hourly")
    cube = Cube(
        _era5_data("hourly"),
        long_name="snow",
        var_name="snow",
        units="unknown",
        dim_coords_and_dims=[
            (time, 0),
            (_era5_latitude(), 1),
            (_era5_longitude(), 2),
        ],
    )
    return CubeList([cube])


def prsn_cmor_e1hr():
    cmor_table = CMOR_TABLES["native6"]
    vardef = cmor_table.get_variable("E1hr", "prsn")
    time = _cmor_time("E1hr", shifted=True, bounds=True)
    data = _cmor_data("E1hr") * 1000 / 3600.0
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


def ptype_era5_hourly():
    time = _era5_time("hourly")
    cube = Cube(
        _era5_data("hourly"),
        long_name="snow",
        var_name="snow",
        units="unknown",
        dim_coords_and_dims=[
            (time, 0),
            (_era5_latitude(), 1),
            (_era5_longitude(), 2),
        ],
    )
    return CubeList([cube])


def ptype_cmor_e1hr():
    cmor_table = CMOR_TABLES["native6"]
    vardef = cmor_table.get_variable("E1hr", "ptype")
    time = _cmor_time("E1hr", shifted=False, bounds=True)
    data = _cmor_data("E1hr")
    cube = Cube(
        data.astype("float32"),
        long_name=vardef.long_name,
        var_name=vardef.short_name,
        units=1,
        dim_coords_and_dims=[
            (time, 0),
            (_cmor_latitude(), 1),
            (_cmor_longitude(), 2),
        ],
        attributes={"comment": COMMENT},
    )
    cube.coord("latitude").long_name = "latitude"
    cube.coord("longitude").long_name = "longitude"
    return CubeList([cube])


def rlds_era5_hourly():
    time = _era5_time("hourly")
    cube = Cube(
        _era5_data("hourly"),
        long_name="surface thermal radiation downwards",
        var_name="ssrd",
        units="J m**-2",
        dim_coords_and_dims=[
            (time, 0),
            (_era5_latitude(), 1),
            (_era5_longitude(), 2),
        ],
    )
    return CubeList([cube])


def rlds_cmor_e1hr():
    cmor_table = CMOR_TABLES["native6"]
    vardef = cmor_table.get_variable("E1hr", "rlds")
    time = _cmor_time("E1hr", shifted=True, bounds=True)
    data = _cmor_data("E1hr") / 3600
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
        attributes={"comment": COMMENT, "positive": "down"},
    )
    return CubeList([cube])


def rlns_era5_hourly():
    freq = "hourly"
    cube = Cube(
        _era5_data(freq),
        long_name=None,
        var_name=None,
        units="J m**-2",
        dim_coords_and_dims=[
            (_era5_time(freq), 0),
            (_era5_latitude(), 1),
            (_era5_longitude(), 2),
        ],
    )
    return CubeList([cube])


def rlns_cmor_e1hr():
    mip = "E1hr"
    short_name = "rlns"
    cmor_table = CMOR_TABLES["native6"]
    vardef = cmor_table.get_variable(mip, short_name)
    time = _cmor_time(mip, shifted=True, bounds=True)
    data = _cmor_data(mip) / 3600
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
        attributes={"comment": COMMENT, "positive": "down"},
    )
    cube.coord("latitude").long_name = "latitude"  # from custom table
    cube.coord("longitude").long_name = "longitude"  # from custom table
    return CubeList([cube])


def rlus_era5_hourly():
    freq = "hourly"
    cube = Cube(
        _era5_data(freq),
        long_name=None,
        var_name=None,
        units="J m**-2",
        dim_coords_and_dims=[
            (_era5_time(freq), 0),
            (_era5_latitude(), 1),
            (_era5_longitude(), 2),
        ],
    )
    return CubeList([cube])


def rlus_cmor_e1hr():
    mip = "E1hr"
    short_name = "rlus"
    cmor_table = CMOR_TABLES["native6"]
    vardef = cmor_table.get_variable(mip, short_name)
    time = _cmor_time(mip, shifted=True, bounds=True)
    data = _cmor_data(mip) / 3600
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
        attributes={"comment": COMMENT, "positive": "up"},
    )
    return CubeList([cube])


def rlut_era5_monthly():
    freq = "monthly"
    cube = Cube(
        _era5_data(freq),
        long_name=None,
        var_name=None,
        units="W m**-2",
        dim_coords_and_dims=[
            (_era5_time(freq), 0),
            (_era5_latitude(), 1),
            (_era5_longitude(), 2),
        ],
    )
    return CubeList([cube])


def rlut_cmor_amon():
    mip = "Amon"
    short_name = "rlut"
    cmor_table = CMOR_TABLES["native6"]
    vardef = cmor_table.get_variable(mip, short_name)
    time = _cmor_time(mip, shifted=True, bounds=True)
    data = -_cmor_data(mip)
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
        attributes={"comment": COMMENT, "positive": "up"},
    )
    return CubeList([cube])


def rlutcs_era5_monthly():
    freq = "monthly"
    cube = Cube(
        _era5_data(freq),
        long_name=None,
        var_name=None,
        units="W m**-2",
        dim_coords_and_dims=[
            (_era5_time(freq), 0),
            (_era5_latitude(), 1),
            (_era5_longitude(), 2),
        ],
    )
    return CubeList([cube])


def rlutcs_cmor_amon():
    mip = "Amon"
    short_name = "rlutcs"
    cmor_table = CMOR_TABLES["native6"]
    vardef = cmor_table.get_variable(mip, short_name)
    time = _cmor_time(mip, shifted=True, bounds=True)
    data = -_cmor_data(mip)
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
        attributes={"comment": COMMENT, "positive": "up"},
    )
    return CubeList([cube])


def rls_era5_hourly():
    time = _era5_time("hourly")
    cube = Cube(
        _era5_data("hourly"),
        long_name="runoff",
        var_name="runoff",
        units="W m-2",
        dim_coords_and_dims=[
            (time, 0),
            (_era5_latitude(), 1),
            (_era5_longitude(), 2),
        ],
    )
    return CubeList([cube])


def rls_cmor_e1hr():
    cmor_table = CMOR_TABLES["native6"]
    vardef = cmor_table.get_variable("E1hr", "rls")
    time = _cmor_time("E1hr", shifted=True, bounds=True)
    data = _cmor_data("E1hr")
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
        attributes={"comment": COMMENT, "positive": "down"},
    )
    return CubeList([cube])


def rsds_era5_hourly():
    time = _era5_time("hourly")
    cube = Cube(
        _era5_data("hourly"),
        long_name="solar_radiation_downwards",
        var_name="rlwd",
        units="J m**-2",
        dim_coords_and_dims=[
            (time, 0),
            (_era5_latitude(), 1),
            (_era5_longitude(), 2),
        ],
    )
    return CubeList([cube])


def rsds_cmor_e1hr():
    cmor_table = CMOR_TABLES["native6"]
    vardef = cmor_table.get_variable("E1hr", "rsds")
    time = _cmor_time("E1hr", shifted=True, bounds=True)
    data = _cmor_data("E1hr") / 3600
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
        attributes={"comment": COMMENT, "positive": "down"},
    )
    return CubeList([cube])


def rsns_era5_hourly():
    freq = "hourly"
    cube = Cube(
        _era5_data(freq),
        long_name=None,
        var_name=None,
        units="J m**-2",
        dim_coords_and_dims=[
            (_era5_time(freq), 0),
            (_era5_latitude(), 1),
            (_era5_longitude(), 2),
        ],
    )
    return CubeList([cube])


def rsns_cmor_e1hr():
    mip = "E1hr"
    short_name = "rsns"
    cmor_table = CMOR_TABLES["native6"]
    vardef = cmor_table.get_variable(mip, short_name)
    time = _cmor_time(mip, shifted=True, bounds=True)
    data = _cmor_data(mip) / 3600
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
        attributes={"comment": COMMENT, "positive": "down"},
    )
    cube.coord("latitude").long_name = "latitude"  # from custom table
    cube.coord("longitude").long_name = "longitude"  # from custom table
    return CubeList([cube])


def rsus_era5_hourly():
    freq = "hourly"
    cube = Cube(
        _era5_data(freq),
        long_name=None,
        var_name=None,
        units="J m**-2",
        dim_coords_and_dims=[
            (_era5_time(freq), 0),
            (_era5_latitude(), 1),
            (_era5_longitude(), 2),
        ],
    )
    return CubeList([cube])


def rsus_cmor_e1hr():
    mip = "E1hr"
    short_name = "rsus"
    cmor_table = CMOR_TABLES["native6"]
    vardef = cmor_table.get_variable(mip, short_name)
    time = _cmor_time(mip, shifted=True, bounds=True)
    data = _cmor_data(mip) / 3600
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
        attributes={"comment": COMMENT, "positive": "up"},
    )
    return CubeList([cube])


def rsdt_era5_hourly():
    time = _era5_time("hourly")
    cube = Cube(
        _era5_data("hourly"),
        long_name="thermal_radiation_downwards",
        var_name="strd",
        units="J m**-2",
        dim_coords_and_dims=[
            (time, 0),
            (_era5_latitude(), 1),
            (_era5_longitude(), 2),
        ],
    )
    return CubeList([cube])


def rsdt_cmor_e1hr():
    cmor_table = CMOR_TABLES["native6"]
    vardef = cmor_table.get_variable("E1hr", "rsdt")
    time = _cmor_time("E1hr", shifted=True, bounds=True)
    data = _cmor_data("E1hr") / 3600
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
        attributes={"comment": COMMENT, "positive": "down"},
    )
    return CubeList([cube])


def rss_era5_hourly():
    time = _era5_time("hourly")
    cube = Cube(
        _era5_data("hourly"),
        long_name="net_solar_radiation",
        var_name="ssr",
        units="J m**-2",
        dim_coords_and_dims=[
            (time, 0),
            (_era5_latitude(), 1),
            (_era5_longitude(), 2),
        ],
    )
    return CubeList([cube])


def rss_cmor_e1hr():
    cmor_table = CMOR_TABLES["native6"]
    vardef = cmor_table.get_variable("E1hr", "rss")
    time = _cmor_time("E1hr", shifted=True, bounds=True)
    data = _cmor_data("E1hr") / 3600
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
        attributes={"comment": COMMENT, "positive": "down"},
    )
    return CubeList([cube])


def sftlf_era5():
    cube = Cube(
        np.ones((3, 3)),
        long_name=None,
        var_name=None,
        units="unknown",
        dim_coords_and_dims=[
            (_era5_latitude(), 0),
            (_era5_longitude(), 1),
        ],
    )
    return CubeList([cube])


def sftlf_cmor_fx():
    cmor_table = CMOR_TABLES["native6"]
    vardef = cmor_table.get_variable("fx", "sftlf")
    cube = Cube(
        np.ones((3, 3)).astype("float32") * 100.0,
        long_name=vardef.long_name,
        var_name=vardef.short_name,
        standard_name=vardef.standard_name,
        units=Unit(vardef.units),
        dim_coords_and_dims=[(_cmor_latitude(), 0), (_cmor_longitude(), 1)],
        attributes={"comment": COMMENT},
    )
    return CubeList([cube])


def tas_era5_hourly():
    time = _era5_time("hourly")
    cube = Cube(
        _era5_data("hourly"),
        long_name="2m_temperature",
        var_name="t2m",
        units="K",
        dim_coords_and_dims=[
            (time, 0),
            (_era5_latitude(), 1),
            (_era5_longitude(), 2),
        ],
    )
    return CubeList([cube])


def tas_cmor_e1hr():
    cmor_table = CMOR_TABLES["native6"]
    vardef = cmor_table.get_variable("E1hr", "tas")
    time = _cmor_time("E1hr")
    data = _cmor_data("E1hr")
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
    cube.add_aux_coord(_cmor_aux_height(2.0))
    return CubeList([cube])


def tas_era5_monthly():
    time = _era5_time("monthly")
    cube = Cube(
        _era5_data("monthly"),
        long_name="2m_temperature",
        var_name="t2m",
        units="K",
        dim_coords_and_dims=[
            (time, 0),
            (_era5_latitude(), 1),
            (_era5_longitude(), 2),
        ],
    )
    return CubeList([cube])


def tas_cmor_amon():
    cmor_table = CMOR_TABLES["native6"]
    vardef = cmor_table.get_variable("Amon", "tas")
    time = _cmor_time("Amon", bounds=True)
    data = _cmor_data("Amon")
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
    cube.add_aux_coord(_cmor_aux_height(2.0))
    return CubeList([cube])


def toz_era5_monthly():
    cube = era5_2d("monthly")[0]
    cube.data = cube.data.astype("float32")
    cube.data *= 2.1415
    return CubeList([cube])


def zg_era5_monthly():
    time = _era5_time("monthly")
    data = np.ones((3, 2, 3, 3))
    cube = Cube(
        data,
        long_name="geopotential height",
        var_name="zg",
        units="m**2 s**-2",
        dim_coords_and_dims=[
            (time, 0),
            (_era5_plev(), 1),
            (_era5_latitude(), 2),
            (_era5_longitude(), 3),
        ],
    )
    return CubeList([cube])


def zg_cmor_amon():
    cmor_table = CMOR_TABLES["native6"]
    vardef = cmor_table.get_variable("Amon", "zg")
    time = _cmor_time("Amon", bounds=True)
    data = np.ones((3, 2, 3, 3))
    data = data / 9.80665
    cube = Cube(
        data.astype("float32"),
        long_name=vardef.long_name,
        var_name=vardef.short_name,
        standard_name=vardef.standard_name,
        units=Unit(vardef.units),
        dim_coords_and_dims=[
            (time, 0),
            (_cmor_plev(), 1),
            (_cmor_latitude(), 2),
            (_cmor_longitude(), 3),
        ],
        attributes={"comment": COMMENT},
    )
    return CubeList([cube])


def tasmax_era5_hourly():
    time = _era5_time("hourly")
    cube = Cube(
        _era5_data("hourly"),
        long_name="maximum 2m temperature",
        var_name="mx2t",
        units="K",
        dim_coords_and_dims=[
            (time, 0),
            (_era5_latitude(), 1),
            (_era5_longitude(), 2),
        ],
    )
    return CubeList([cube])


def tasmax_cmor_e1hr():
    cmor_table = CMOR_TABLES["native6"]
    vardef = cmor_table.get_variable("E1hr", "tasmax")
    time = _cmor_time("E1hr", shifted=True, bounds=True)
    data = _cmor_data("E1hr")
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
    cube.add_aux_coord(_cmor_aux_height(2.0))
    return CubeList([cube])


def tasmin_era5_hourly():
    time = _era5_time("hourly")
    cube = Cube(
        _era5_data("hourly"),
        long_name="minimum 2m temperature",
        var_name="mn2t",
        units="K",
        dim_coords_and_dims=[
            (time, 0),
            (_era5_latitude(), 1),
            (_era5_longitude(), 2),
        ],
    )
    return CubeList([cube])


def tasmin_cmor_e1hr():
    cmor_table = CMOR_TABLES["native6"]
    vardef = cmor_table.get_variable("E1hr", "tasmin")
    time = _cmor_time("E1hr", shifted=True, bounds=True)
    data = _cmor_data("E1hr")
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
    cube.add_aux_coord(_cmor_aux_height(2.0))
    return CubeList([cube])


def uas_era5_hourly():
    time = _era5_time("hourly")
    cube = Cube(
        _era5_data("hourly"),
        long_name="10m_u_component_of_wind",
        var_name="u10",
        units="m s-1",
        dim_coords_and_dims=[
            (time, 0),
            (_era5_latitude(), 1),
            (_era5_longitude(), 2),
        ],
    )
    return CubeList([cube])


def uas_cmor_e1hr():
    cmor_table = CMOR_TABLES["native6"]
    vardef = cmor_table.get_variable("E1hr", "uas")
    time = _cmor_time("E1hr")
    data = _cmor_data("E1hr")
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
    cube.add_aux_coord(_cmor_aux_height(10.0))
    return CubeList([cube])


def vas_era5_hourly():
    time = _era5_time("hourly")
    cube = Cube(
        _era5_data("hourly"),
        long_name="10m_v_component_of_wind",
        var_name="v10",
        units="m s-1",
        dim_coords_and_dims=[
            (time, 0),
            (_era5_latitude(), 1),
            (_era5_longitude(), 2),
        ],
    )
    return CubeList([cube])


def vas_cmor_e1hr():
    cmor_table = CMOR_TABLES["native6"]
    vardef = cmor_table.get_variable("E1hr", "vas")
    time = _cmor_time("E1hr")
    data = _cmor_data("E1hr")
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
    cube.add_aux_coord(_cmor_aux_height(10.0))
    return CubeList([cube])


VARIABLES = [
    pytest.param(a, b, c, d, id=c + "_" + d)
    for (a, b, c, d) in [
        (era5_2d("daily"), cmor_2d("Eday", "albsn"), "albsn", "Eday"),
        (cl_era5_monthly(), cl_cmor_amon(), "cl", "Amon"),
        (era5_3d("monthly"), cmor_3d("Amon", "cli"), "cli", "Amon"),
        (clt_era5_hourly(), clt_cmor_e1hr(), "clt", "E1hr"),
        (era5_3d("monthly"), cmor_3d("Amon", "clw"), "clw", "Amon"),
        (evspsbl_era5_hourly(), evspsbl_cmor_e1hr(), "evspsbl", "E1hr"),
        (
            evspsblpot_era5_hourly(),
            evspsblpot_cmor_e1hr(),
            "evspsblpot",
            "E1hr",
        ),
        (era5_3d("monthly"), cmor_3d("Amon", "hus"), "hus", "Amon"),
        (mrro_era5_hourly(), mrro_cmor_e1hr(), "mrro", "E1hr"),
        (o3_era5_monthly(), cmor_3d("Amon", "o3"), "o3", "Amon"),
        (orog_era5_hourly(), orog_cmor_fx(), "orog", "fx"),
        (pr_era5_monthly(), pr_cmor_amon(), "pr", "Amon"),
        (pr_era5_hourly(), pr_cmor_e1hr(), "pr", "E1hr"),
        (prsn_era5_hourly(), prsn_cmor_e1hr(), "prsn", "E1hr"),
        (era5_2d("monthly"), cmor_2d("Amon", "prw"), "prw", "Amon"),
        (era5_2d("monthly"), cmor_2d("Amon", "ps"), "ps", "Amon"),
        (ptype_era5_hourly(), ptype_cmor_e1hr(), "ptype", "E1hr"),
        (
            era5_3d("monthly"),
            cmor_3d("Emon", "rainmxrat27"),
            "rainmxrat27",
            "Emon",
        ),
        (rlds_era5_hourly(), rlds_cmor_e1hr(), "rlds", "E1hr"),
        (rlns_era5_hourly(), rlns_cmor_e1hr(), "rlns", "E1hr"),
        (rlus_era5_hourly(), rlus_cmor_e1hr(), "rlus", "E1hr"),
        (rlut_era5_monthly(), rlut_cmor_amon(), "rlut", "Amon"),
        (rlutcs_era5_monthly(), rlutcs_cmor_amon(), "rlutcs", "Amon"),
        (rls_era5_hourly(), rls_cmor_e1hr(), "rls", "E1hr"),
        (rsds_era5_hourly(), rsds_cmor_e1hr(), "rsds", "E1hr"),
        (rsns_era5_hourly(), rsns_cmor_e1hr(), "rsns", "E1hr"),
        (rsus_era5_hourly(), rsus_cmor_e1hr(), "rsus", "E1hr"),
        (rsdt_era5_hourly(), rsdt_cmor_e1hr(), "rsdt", "E1hr"),
        (rss_era5_hourly(), rss_cmor_e1hr(), "rss", "E1hr"),
        (sftlf_era5(), sftlf_cmor_fx(), "sftlf", "fx"),
        (
            era5_3d("monthly"),
            cmor_3d("Emon", "snowmxrat27"),
            "snowmxrat27",
            "Emon",
        ),
        (tas_era5_hourly(), tas_cmor_e1hr(), "tas", "E1hr"),
        (tas_era5_monthly(), tas_cmor_amon(), "tas", "Amon"),
        (tasmax_era5_hourly(), tasmax_cmor_e1hr(), "tasmax", "E1hr"),
        (tasmin_era5_hourly(), tasmin_cmor_e1hr(), "tasmin", "E1hr"),
        (toz_era5_monthly(), cmor_2d("AERmon", "toz"), "toz", "AERmon"),
        (uas_era5_hourly(), uas_cmor_e1hr(), "uas", "E1hr"),
        (vas_era5_hourly(), vas_cmor_e1hr(), "vas", "E1hr"),
        (zg_era5_monthly(), zg_cmor_amon(), "zg", "Amon"),
    ]
]


@pytest.mark.parametrize(("era5_cubes", "cmor_cubes", "var", "mip"), VARIABLES)
def test_cmorization(era5_cubes, cmor_cubes, var, mip):
    """Verify that cmorization results in the expected target cube."""
    fixed_cubes = fix_metadata(era5_cubes, var, "native6", "era5", mip)

    assert len(fixed_cubes) == 1
    fixed_cube = fixed_cubes[0]
    cmor_cube = cmor_cubes[0]

    # Test that CMOR checks are passing
    fixed_cubes = cmor_check_metadata(fixed_cube, "native6", mip, var)

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
        [0.0, 31.0],
        standard_name="time",
        units="days since 1950-01-01",
    )
    lat = AuxCoord(
        [1.0, 1.0, -1.0, -1.0],
        standard_name="latitude",
        units="degrees_north",
    )
    lon = AuxCoord(
        [179.0, 180.0, 180.0, 179.0],
        standard_name="longitude",
        units="degrees_east",
    )
    cube = Cube(
        da.from_array([[0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 0.0]]),
        standard_name="air_temperature",
        units="K",
        dim_coords_and_dims=[(time, 0)],
        aux_coords_and_dims=[(lat, 1), (lon, 1)],
        attributes={"GRIB_PARAM": (1, 1)},
    )
    return CubeList([cube])


def test_unstructured_grid(unstructured_grid_cubes):
    """Test processing unstructured data."""
    fixed_cubes = fix_metadata(
        unstructured_grid_cubes,
        "tas",
        "native6",
        "era5",
        "Amon",
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

    assert fixed_cube.attributes["GRIB_PARAM"] == "(1, 1)"
