"""Check requested coverage using lazy cubes and actual time coordinates."""

from __future__ import annotations

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from cf_units import Unit
from cftime import datetime as cf_datetime
from iris.coords import DimCoord
from iris.cube import Cube

from esmvalcore._recipe import check
from esmvalcore._time import _parse_timerange
from esmvalcore.dataset import Dataset
from esmvalcore.exceptions import InputFilesNotFound


@pytest.mark.parametrize("calendar", ["standard", "360_day", "noleap"])
@pytest.mark.parametrize(
    ("frequency", "step", "timerange"),
    [
        ("yr", "YS", "2000/2001"),
        ("mon", "MS", "2000/2001"),
        ("day", "D", "200001/200002"),
        ("3hrPt", "3h", "200001/200002"),
    ],
)
@pytest.mark.parametrize("missing_end", [None, "start", "end"])
def test_loaded_coverage(
    calendar: str,
    frequency: str,
    step: str,
    timerange: str,
    missing_end: str | None,
) -> None:
    """Check endpoints at sampling resolution, preserving field laziness."""
    dataset = Dataset(frequency=frequency, timerange=timerange)
    end = "2002-01-01" if frequency in {"mon", "yr"} else "2000-03-01"
    dates = xr.date_range(
        "2000-01-01",
        end,
        freq=step,
        inclusive="left",
        calendar=calendar,
        use_cftime=True,
    )
    if missing_end == "start":
        dates = dates[1:]
    elif missing_end == "end":
        dates = dates[:-1]
    unit = Unit("days since 2000-01-01", calendar=calendar)
    time = DimCoord(unit.date2num(dates), standard_name="time", units=unit)
    cube = Cube(
        da.zeros(len(dates), chunks=10),
        dim_coords_and_dims=[(time, 0)],
    )
    if missing_end:
        with pytest.raises(InputFilesNotFound, match="do not cover requested"):
            check.loaded_data_availability(dataset, cube)
    else:
        check.loaded_data_availability(dataset, cube)
    assert cube.has_lazy_data()


def test_loaded_coverage_with_duration() -> None:
    """Duration ranges must resolve exactly as they do when clipping."""
    dataset = Dataset(frequency="mon", timerange="P1Y/2001")
    unit = Unit("days since 2001-01-01", calendar="360_day")
    time = DimCoord(
        [15 + 30 * month for month in range(12)],
        standard_name="time",
        units=unit,
    )
    cube = Cube(da.zeros(12, chunks=12), dim_coords_and_dims=[(time, 0)])
    check.loaded_data_availability(dataset, cube)
    assert cube.has_lazy_data()


def test_loaded_coverage_with_missing_internal_year() -> None:
    """The endpoints do not establish that every intervening year exists."""
    dataset = Dataset(frequency="yr", timerange="2000/2002")
    unit = Unit("days since 2000-01-01", calendar="360_day")
    time = DimCoord([180, 900], standard_name="time", units=unit)
    cube = Cube(da.zeros(2, chunks=2), dim_coords_and_dims=[(time, 0)])
    with pytest.raises(InputFilesNotFound, match="do not cover requested"):
        check.loaded_data_availability(dataset, cube)
    assert cube.has_lazy_data()


@pytest.mark.parametrize(
    ("bounds", "timerange", "complete"),
    [
        ([[0, 3600], [3600, 7200]], "2000/2019", True),
        ([[0, 3600]], "2000/2019", False),
        ([[0, 3600], [7200, 10800]], "2000/2029", False),
        ([[-3600, 0], [0, 3600]], "2000/2009", True),
    ],
)
def test_decadal_coverage_uses_actual_bounds(
    bounds: list[list[int]],
    timerange: str,
    complete: bool,
) -> None:
    """Multi-year means cover their intervals, including years without points."""
    dataset = Dataset(frequency="dec", timerange=timerange)
    unit = Unit("days since 2000-01-01", calendar="360_day")
    time = DimCoord(
        [(lower + upper) / 2 for lower, upper in bounds],
        bounds=bounds,
        standard_name="time",
        units=unit,
    )
    cube = Cube(
        da.zeros(len(bounds), chunks=1),
        dim_coords_and_dims=[(time, 0)],
    )
    if complete:
        check.loaded_data_availability(dataset, cube)
    else:
        with pytest.raises(InputFilesNotFound, match="do not cover requested"):
            check.loaded_data_availability(dataset, cube)
    assert cube.has_lazy_data()


@pytest.mark.parametrize(
    ("timerange", "complete"),
    [("2000/2019", True), ("1999/2019", False), ("2000/2020", False)],
)
def test_climatological_coverage_uses_bounds(
    timerange: str,
    complete: bool,
) -> None:
    """Climatological samples need not have a point in every averaged year."""
    dataset = Dataset(frequency="monC", timerange=timerange)
    unit = Unit("days since 2000-01-01", calendar="360_day")
    time = DimCoord(
        [15 + 30 * month for month in range(12)],
        bounds=[
            [30 * month, 19 * 360 + 30 * (month + 1)] for month in range(12)
        ],
        standard_name="time",
        units=unit,
        climatological=True,
    )
    cube = Cube(da.zeros(12, chunks=12), dim_coords_and_dims=[(time, 0)])
    if complete:
        check.loaded_data_availability(dataset, cube)
    else:
        with pytest.raises(InputFilesNotFound, match="do not cover requested"):
            check.loaded_data_availability(dataset, cube)
    assert cube.has_lazy_data()


@pytest.mark.parametrize(
    "facets",
    [{"frequency": "fx", "timerange": "2000/2001"}, {"frequency": "mon"}],
)
def test_no_requested_time_coverage(facets: dict[str, str]) -> None:
    """Fixed fields and unrestricted requests require no temporal selection."""
    cube = Cube(da.zeros((2, 2), chunks=2))
    check.loaded_data_availability(Dataset(**facets), cube)
    assert cube.has_lazy_data()


def test_subhourly_cadence_cannot_be_inferred() -> None:
    """Irregular unbounded point samples cannot establish requested coverage."""
    dataset = Dataset(frequency="subhrPt", timerange="200001/200001")
    unit = Unit("minutes since 2000-01-01", calendar="360_day")
    time = DimCoord([0, 30, 90], standard_name="time", units=unit)
    cube = Cube(da.zeros(3, chunks=3), dim_coords_and_dims=[(time, 0)])
    with pytest.raises(InputFilesNotFound, match="Cannot validate"):
        check.loaded_data_availability(dataset, cube)
    assert cube.has_lazy_data()


def test_time_range_requires_date() -> None:
    """Two durations cannot establish the reference date for a selection."""
    with pytest.raises(TypeError, match="at least one date"):
        _parse_timerange("P1Y/P1Y")


@pytest.mark.parametrize("calendar", ["standard", "360_day"])
@pytest.mark.parametrize("origin", ["0001", "1850", "2000"])
@pytest.mark.parametrize("samples", [47, 48])
def test_subhourly_coverage_tolerates_cf_roundoff(
    calendar: str,
    origin: str,
    samples: int,
) -> None:
    """A distant float64 epoch must not make a regular cadence look irregular."""
    unit = Unit(f"days since {origin}-01-01", calendar=calendar)
    start = unit.date2num(cf_datetime(2000, 1, 1, calendar=calendar))
    time = DimCoord(
        start + np.arange(samples, dtype="float64") / 48,
        standard_name="time",
        units=unit,
    )
    time.convert_units(Unit("days since 1850-01-01", calendar=calendar))
    original_points = time.points.copy()
    cube = Cube(
        da.zeros(samples, chunks=samples),
        dim_coords_and_dims=[(time, 0)],
    )
    dataset = Dataset(
        frequency="subhrPt",
        timerange="20000101T000000/20000101T235959",
    )
    if samples == 48:
        check.loaded_data_availability(dataset, cube)
    else:
        with pytest.raises(InputFilesNotFound, match="do not cover"):
            check.loaded_data_availability(dataset, cube)
    np.testing.assert_array_equal(time.points, original_points)
    assert cube.has_lazy_data()


@pytest.mark.parametrize("missing_end", [None, "start", "end"])
def test_subhourly_coverage_preserves_sampling_phase(
    missing_end: str | None,
) -> None:
    """A 25-minute cadence does not restart at each midnight."""
    unit = Unit("minutes since 2000-01-01", calendar="360_day")
    points = np.arange(0, 5 * 1440, 25, dtype=float)
    points = points[(points >= 2 * 1440) & (points < 3 * 1440)]
    if missing_end == "start":
        points = points[1:]
    elif missing_end == "end":
        points = points[:-1]
    time = DimCoord(points, standard_name="time", units=unit)
    cube = Cube(
        da.zeros(len(points), chunks=100),
        dim_coords_and_dims=[(time, 0)],
    )
    dataset = Dataset(
        frequency="subhrPt",
        timerange="20000103T000000/20000103T235959",
    )
    if missing_end:
        with pytest.raises(InputFilesNotFound, match="do not cover"):
            check.loaded_data_availability(dataset, cube)
    else:
        check.loaded_data_availability(dataset, cube)
    assert cube.has_lazy_data()
