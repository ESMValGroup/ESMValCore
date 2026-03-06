"""Example implementation of a configurable fixer for ESA CCI data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Self

import cf_xarray.units  # noqa: F401 # Needed to support cf-units with pint
import cftime
import numpy as np
import pint
import xarray as xr

if TYPE_CHECKING:
    import dask.array as da

# ureg = UnitRegistry()
# ureg.define("degrees_north = [latitude]")
# ureg.define("degrees_east = [longitude]")


def _convert_units(
    data: da.Array,
    src_units: str | None,
    tgt_units: str | None,
) -> da.Array:
    """Convert units.

    Parameters
    ----------
    data:
        The data to convert.
    src_units:
        The source units.
    tgt_units:
        The target units. If `None`, no conversion is performed.

    Returns
    -------
    :
        The converted data.

    """
    if tgt_units is None:
        return data
    if src_units is None:
        msg = f"Unable to convert unknown units to {tgt_units}."
        raise ValueError(msg)
    return pint.Quantity(data, src_units).to(tgt_units).magnitude


@dataclass
class CoordinateBounds:
    """Definition of a coordinate bounds variable."""

    name: str
    dims: tuple[str, ...]


@dataclass
class Coordinate:
    """Definition of a coordinate variable."""

    name: str
    dtype: str | None
    dims: tuple[str, ...] | None
    attrs: dict[str, str] | None
    bounds: CoordinateBounds | None

    @classmethod
    def from_cmor_table(cls, path: Path, entry: str) -> Self:
        """Load a definition from the CMIP7 CMOR tables."""
        spec = json.loads(
            (path / "CMIP7_coordinate.json").read_text(encoding="utf-8"),
        )["axis_entry"][entry]
        dtype = "float64" if spec.get("dtype", "") == "double" else None
        attrs: tuple[str, ...] = (
            "standard_name",
            "long_name",
            "axis",
            "positive",
        )
        if spec["standard_name"] != "time":
            # The time units are incomplete in the CMOR tables.
            attrs = (*attrs[:2], "units", *attrs[2:])
        dims = (
            (spec["out_name"],) if spec["axis"] and not spec["value"] else None
        )
        if spec["must_have_bounds"] == "yes":
            bounds = CoordinateBounds(
                name=f"{spec['out_name']}_bnds",
                dims=(
                    *dims,
                    "bnds",
                )
                if dims
                else ("bnds",),
            )
        else:
            bounds = None
        return cls(
            name=spec["out_name"],
            dtype=dtype,
            dims=dims,
            attrs={k: spec[k] for k in attrs if spec[k]},
            bounds=bounds,
        )

    def _copy_time_encoding(
        self,
        src: xr.DataArray,
        tgt: xr.DataArray,
    ) -> None:
        """Copy the time encoding from `src` to `tgt` if it exists."""
        time_encoding_keys = ("units", "calendar", "dtype")
        if set(time_encoding_keys).issubset(src.encoding):
            for key in time_encoding_keys:
                tgt.encoding[key] = src.encoding[key]
                tgt.attrs.pop(key, None)

    def to_dataarray(
        self,
        ds: xr.Dataset,
        dim_map: dict[str, str] | None = None,
        variable_map: dict[str, str] | None = None,
    ) -> xr.DataArray:
        """Create a coordinate using the data from `ds`."""
        attrs = dict(self.attrs or {})
        if self.bounds:
            attrs["bounds"] = self.bounds.name
        variable_map = variable_map or {}
        original_coord = ds[variable_map.get(self.name, self.name)]
        if dim_map is None:
            dim_map = {d: d for d in original_coord.dims}
        if self.dims is not None:
            order = [dim_map.get(d, d) for d in self.dims]
            data = original_coord.transpose(*order).data
        else:
            data = original_coord.data
        data = _convert_units(
            data,
            original_coord.attrs.get("units"),
            attrs.get("units"),
        )
        if self.dtype is not None:
            data = data.astype(np.dtype(self.dtype))
        coord = xr.DataArray(
            data=data,
            name=self.name,
            dims=self.dims,
            attrs=attrs,
        )
        self._copy_time_encoding(original_coord, coord)
        return coord

    def to_bounds_dataarray(
        self,
        ds: xr.Dataset,
        dim_map: dict[str, str] | None = None,
        variable_map: dict[str, str] | None = None,
    ) -> xr.DataArray | None:
        """Create coordinate bounds using the data from `ds`."""
        if not self.bounds:
            return None
        variable_map = variable_map or {}
        original_coord = ds[variable_map.get(self.name, self.name)]
        original_bounds = ds[variable_map.get(self.name, self.name)]
        if dim_map is None:
            dim_map = {d: d for d in original_bounds.dims}
        order = [dim_map.get(d, d) for d in self.dims]
        data = original_bounds.transpose(*order).data
        data = _convert_units(
            data,
            original_coord.attrs.get("units"),
            (self.attrs or {}).get("units"),
        )
        if self.dtype is not None:
            data = data.astype(np.dtype(self.dtype))
        bounds = xr.DataArray(
            data=data,
            name=self.name,
            dims=self.dims,
        )
        self._copy_time_encoding(original_coord, bounds)
        return bounds


@dataclass
class Variable:
    """Definition of a physical quantity."""

    name: str
    dtype: str | None
    dims: tuple[str, ...] | None
    coords: tuple[Coordinate, ...] | None
    attrs: dict[str, str] | None

    @classmethod
    def from_cmor_table(cls, path: Path, table_id: str, entry: str) -> Self:
        """Load a definition from the CMIP7 CMOR tables."""
        spec = json.loads(
            (path / f"CMIP7_{table_id}.json").read_text(encoding="utf-8"),
        )["variable_entry"][entry]

        attrs = (
            "standard_name",
            "long_name",
            "units",
            "cell_methods",
            "cell_measures",
            "positive",
        )
        coords = tuple(
            Coordinate.from_cmor_table(path, d)
            for d in spec["dimensions"][::-1]
        )
        dims = tuple(d.name for d in coords if d.dims) if coords else None
        return cls(
            name=spec["out_name"],
            dtype="float32",
            dims=dims,
            coords=coords,
            attrs={k: spec[k] for k in attrs if spec[k]},
        )

    def to_dataset(
        self,
        ds: xr.Dataset,
        dim_map: dict[str, str] | None = None,
        variable_map: dict[str, str] | None = None,
    ) -> xr.Dataset:
        """Create a dataset using the data from `ds`."""
        variable_map = variable_map or {}
        original_var = ds[variable_map.get(self.name, self.name)]
        if dim_map is None:
            dim_map = {d: d for d in original_var.dims}
        if self.dims is None:
            data = original_var.data
        else:
            order = [dim_map.get(d, d) for d in self.dims]
            data = original_var.transpose(*order).data
        data = _convert_units(
            data,
            original_var.attrs.get("units"),
            self.attrs.get("units"),
        )
        if self.dtype is not None:
            data = data.astype(np.dtype(self.dtype))
        coords = {
            c.name: c.to_dataarray(ds, dim_map, variable_map)
            for c in self.coords or ()
        }
        var = xr.DataArray(
            data=data,
            coords=coords,
            dims=self.dims,
            name=self.name,
            attrs=self.attrs,
        )
        bounds = {
            c.bounds.name: c.to_bounds_dataarray(ds, dim_map, variable_map)
            for c in self.coords or ()
            if c.bounds is not None
        }
        return xr.Dataset({self.name: var}, coords=coords | bounds)


def main():
    cmor_table_path = (
        Path.home() / "src" / "WCRP-CMIP" / "cmip7-cmor-tables" / "tables"
    )
    variable = Variable.from_cmor_table(
        cmor_table_path,
        "atmos",
        "tas_tavg-h2m-hxy-u",
    )
    ds = xr.open_dataset(
        "~/climate_data/CMIP6/CMIP/BCC/BCC-ESM1/historical/r1i1p1f1/Amon/tas/gn/v20181214/tas_Amon_BCC-ESM1_historical_r1i1p1f1_gn_185001-201412.nc",
        chunks={"time": 100},
    )
    print("Original:\n", ds)
    print("Converting..")
    result = variable.to_dataset(ds)
    print("Result:\n", result)
    print("Saving to NetCDF..")
    result.to_netcdf("tas_fixed_big.nc")
    # print(
    #     ds.isel(time=slice(0, 1), lat=slice(0, 2), lon=slice(0, 3)).to_dict()
    # )
    ds2 = xr.Dataset.from_dict(
        {
            "coords": {
                "time": {
                    "dims": ("time",),
                    "attrs": {
                        "bounds": "time_bnds",
                    },
                    "data": [
                        cftime.DatetimeNoLeap(
                            1850,
                            1,
                            16,
                            12,
                            0,
                            0,
                            0,
                            has_year_zero=True,
                        ),
                    ],
                    "encoding": {
                        "units": "days since 1850-01-01 00:00:00",
                        "calendar": "noleap",
                        "dtype": "float64",
                    },
                },
                "lat": {
                    "dims": ("y",),
                    "attrs": {
                        "bounds": "lat_bnds",
                        "units": "degrees_north",
                    },
                    "data": [-87.86379883923263, -85.09652698831736],
                },
                "lon": {
                    "dims": ("x",),
                    "attrs": {
                        "bounds": "lon_bnds",
                        "units": "degrees_east",
                    },
                    "data": [0.0, 2.8125, 5.625],
                },
                "height2m": {
                    "dims": (),
                    "attrs": {
                        "units": "m",
                    },
                    "data": 2.0,
                },
            },
            "attrs": {},
            "dims": {"time": 1, "bnds": 2, "y": 2, "x": 3},
            "data_vars": {
                "time_bnds": {
                    "dims": ("time", "bnds"),
                    "attrs": {},
                    "data": [
                        [
                            cftime.DatetimeNoLeap(
                                1850,
                                1,
                                1,
                                0,
                                0,
                                0,
                                0,
                                has_year_zero=True,
                            ),
                            cftime.DatetimeNoLeap(
                                1850,
                                2,
                                1,
                                0,
                                0,
                                0,
                                0,
                                has_year_zero=True,
                            ),
                        ],
                    ],
                },
                "lat_bnds": {
                    "dims": ("y", "bnds"),
                    "attrs": {},
                    "data": [
                        [-90.0, -86.48016291377499],
                        [-86.48016291377499, -83.70471996810181],
                    ],
                },
                "lon_bnds": {
                    "dims": ("x", "bnds"),
                    "attrs": {},
                    "data": [
                        [-1.40625, 1.40625],
                        [1.40625, 4.21875],
                        [4.21875, 7.03125],
                    ],
                },
                "tas": {
                    "dims": ("time", "y", "x"),
                    "attrs": {
                        "units": "K",
                    },
                    "data": [
                        [
                            [
                                247.4741973876953,
                                247.24557495117188,
                                247.0136260986328,
                            ],
                            [
                                248.81895446777344,
                                248.20892333984375,
                                247.6052703857422,
                            ],
                        ],
                    ],
                },
            },
        },
    )
    print("Original:\n", ds2)
    print("Converting..")
    result = variable.to_dataset(
        ds2,
        dim_map={"lat": "y", "lon": "x"},
        variable_map={"height": "height2m"},
    )
    print("Result:\n", result)
    print("Saving to NetCDF..")
    result.to_netcdf("tas_fixed.nc")


if __name__ == "__main__":
    main()
