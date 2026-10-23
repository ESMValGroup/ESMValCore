"""Tests for :mod:`esmvalcore.io._netcdf_masking`."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import iris
import netCDF4
import numpy as np
import pytest
import xarray as xr

from esmvalcore.io._netcdf_masking import (
    mask_like_netcdf4,
    set_default_calendar,
)
from esmvalcore.io.local import LocalFile
from esmvalcore.iris_helpers import dataset_to_iris

if TYPE_CHECKING:
    from pathlib import Path

FILL = netCDF4.default_fillvals

# Each case: data type, raw values in the file, fill value passed to
# netCDF4.Dataset.createVariable (None: default, False: no filling), and
# other attributes of the variable.
CASES: dict[str, tuple[str, list[Any], Any, dict[str, Any]]] = {
    "default fill f4": ("f4", [1, FILL["f4"]], None, {}),
    "default fill f8": ("f8", [1, FILL["f8"]], None, {}),
    "default fill i2": ("i2", [1, FILL["i2"]], None, {}),
    "default fill u4": ("u4", [1, FILL["u4"]], None, {}),
    "default fill i1": ("i1", [1, FILL["i1"]], None, {}),
    "default fill u1": ("u1", [1, FILL["u1"]], None, {}),
    "default fill i1 no filling": ("i1", [1, FILL["i1"]], False, {}),
    "default fill f4 no filling": ("f4", [1, FILL["f4"]], False, {}),
    "fill value 1e20": ("f4", [1, 1e20, FILL["f4"]], np.float32(1e20), {}),
    "missing_value and default fill": (
        "f4",
        [1, -999, FILL["f4"]],
        None,
        {"missing_value": np.float32(-999)},
    ),
    "valid_min": ("f4", [-1, 5, 11], None, {"valid_min": np.float32(0)}),
    "valid_max": ("f4", [-1, 5, 11], None, {"valid_max": np.float32(10)}),
    "valid_min and valid_max": (
        "f4",
        [-1, 5, 11],
        None,
        {"valid_min": np.float32(0), "valid_max": np.float32(10)},
    ),
    "valid_range": (
        "f4",
        [-1, 0, 5, 10, 11],
        None,
        {"valid_range": np.array([0, 10], "f4")},
    ),
    "valid_range takes precedence": (
        "f4",
        [-4, 5, 11],
        None,
        {
            "valid_range": np.array([0, 10], "f4"),
            "valid_min": np.float32(-5),
        },
    ),
    "valid_range and fill value": (
        "f4",
        [-1, 5, 11, 1e20],
        np.float32(1e20),
        {"valid_range": np.array([0, 10], "f4")},
    ),
    "string valid_min is ignored": (
        "f4",
        [-1, 5],
        None,
        {"valid_min": "0.0"},
    ),
    "unsafe valid_max is ignored": (
        "f4",
        [0.1, 0.2],
        None,
        {"valid_max": np.float64(0.1)},
    ),
    "integer valid_range": (
        "i4",
        [-1, 5, 11],
        None,
        {"valid_range": np.array([0, 10], "i4")},
    ),
    "packed": (
        "i2",
        [-500, -300, 0, 4500, 5000, -32768],
        np.int16(-32768),
        {
            "scale_factor": np.float32(0.01),
            "add_offset": np.float32(273.15),
            "valid_min": np.int16(-300),
            "valid_max": np.int16(4500),
        },
    ),
    "packed with negative scale factor": (
        "i2",
        [-500, 0, 5000, FILL["i2"]],
        None,
        {
            "scale_factor": np.float32(-0.5),
            "valid_range": np.array([-300, 4500], "i2"),
        },
    ),
    "unsigned default fill": (
        "i2",
        [1, FILL["i2"]],
        None,
        {"_Unsigned": "true"},
    ),
    "unsigned": (
        "i2",
        [1, -2, 100, FILL["i2"]],
        None,
        {"_Unsigned": "true", "valid_min": np.int16(2)},
    ),
}


def write(
    path: Path,
    dtype: str,
    values: list,
    fill: Any,  # noqa: ANN401
    attrs: dict,
) -> None:
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("i", len(values))
        variable = dataset.createVariable(
            "x",
            dtype,
            ("i",),
            fill_value=fill,
        )
        variable.set_auto_maskandscale(False)
        variable[:] = np.array(values, dtype)
        for key, value in attrs.items():
            variable.setncattr(key, value)


def test_mask_like_netcdf4_unsigned_max_value(tmp_path: Path) -> None:
    """Known difference: the maximum value of unsigned data is masked.

    The data is converted to Iris cubes as unsigned integers, and Iris masks
    the netCDF default fill value of unsigned types, which is the maximum
    value. The netCDF4 library does not mask any default fill value of data
    with the ``_Unsigned`` attribute.
    """
    path = tmp_path / "test.nc"
    write(path, "i2", [1, -1], None, {"_Unsigned": "true"})
    assert not np.ma.is_masked(iris.load_cube(path).data)

    dataset = xr.open_dataset(path)
    mask_like_netcdf4(dataset)
    (cube,) = dataset_to_iris(dataset)

    assert cube.dtype == np.uint16
    np.testing.assert_array_equal(
        np.ma.getmaskarray(cube.data),
        [False, True],
    )


@pytest.mark.parametrize("case", CASES)
def test_mask_like_netcdf4(tmp_path: Path, case: str) -> None:
    path = tmp_path / "test.nc"
    write(path, *CASES[case])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*cannot be safely cast")
        expected = iris.load_cube(path)
        expected_data = expected.data

    dataset = xr.open_dataset(path, decode_times=False, chunks="auto")
    mask_like_netcdf4(dataset, no_fill=LocalFile(path)._no_fill_variables())
    (cube,) = dataset_to_iris(dataset)

    assert cube.has_lazy_data()
    assert cube.dtype == expected.dtype
    np.testing.assert_array_equal(
        np.ma.getmaskarray(cube.data),
        np.ma.getmaskarray(expected_data),
    )
    np.testing.assert_allclose(
        np.ma.filled(cube.data.astype(np.float64), 0),
        np.ma.filled(expected_data.astype(np.float64), 0),
    )


def test_mask_like_netcdf4_removes_applied_range_attributes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "test.nc"
    write(
        path,
        "f4",
        [1.0],
        None,
        {
            "valid_range": np.array([0, 10], "f4"),
            "valid_min": "ignored",
            "valid_max": np.float32(5),
        },
    )
    dataset = xr.open_dataset(path)
    mask_like_netcdf4(dataset)
    assert "valid_range" not in dataset["x"].attrs
    # These were not applied, because valid_range takes precedence.
    assert dataset["x"].attrs["valid_min"] == "ignored"
    assert dataset["x"].attrs["valid_max"] == 5


def test_mask_like_netcdf4_coordinates(tmp_path: Path) -> None:
    xr.Dataset(
        {"x": ("i", np.array([1, 2, 3], dtype="f4"))},
        coords={
            "i": ("i", [0, 1, 2]),
            "lat": (
                "i",
                [0.0, FILL["f8"], 100.0],
                {"standard_name": "latitude", "valid_max": 90.0},
            ),
        },
    ).to_netcdf(tmp_path / "test.nc", encoding={"lat": {"_FillValue": None}})
    dataset = xr.open_dataset(tmp_path / "test.nc")

    mask_like_netcdf4(dataset)

    np.testing.assert_array_equal(
        np.isnan(dataset["lat"].values),
        [False, True, True],
    )
    # Index coordinates become dimension coordinates, they are not masked.
    assert "_FillValue" not in dataset["i"].encoding


@pytest.mark.parametrize("calendar", [None, "standard", "proleptic_gregorian"])
def test_set_default_calendar(tmp_path: Path, calendar: str | None) -> None:
    path = tmp_path / "test.nc"
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("time", 2)
        time = dataset.createVariable("time", "f8", ("time",))
        time.units = "days since 2000-01-01"
        time.standard_name = "time"
        if calendar is not None:
            time.calendar = calendar
        time[:] = [0.5, 1.5]
        dataset.createVariable("x", "f4", ("time",))[:] = [1, 2]
    expected = iris.load_cube(path).coord("time").units

    dataset = xr.open_dataset(path)
    set_default_calendar(dataset)
    (cube,) = dataset_to_iris(dataset)

    units = cube.coord("time").units
    assert units.calendar == expected.calendar
    assert units.calendar == (calendar or "standard")
