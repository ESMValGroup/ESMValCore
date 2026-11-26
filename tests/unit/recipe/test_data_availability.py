"""Check requested coverage using lazy cubes and actual time coordinates."""

from __future__ import annotations

import dask.array as da
import pytest
import xarray as xr
from cf_units import Unit
from iris.coords import DimCoord
from iris.cube import Cube

from esmvalcore._recipe import check
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
