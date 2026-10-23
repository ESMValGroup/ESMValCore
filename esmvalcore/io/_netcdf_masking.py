"""Mask data opened with :mod:`xarray` like the netCDF4 library does.

The netCDF4 library, and therefore Iris, masks more values than Xarray does
when reading netCDF files. Xarray only masks values equal to the
``_FillValue`` and ``missing_value`` attributes. The netCDF4 library also
masks

- values equal to the netCDF default fill value of a variable without a
  ``_FillValue`` attribute, and
- values outside the range given by the ``valid_range``, or ``valid_min`` and
  ``valid_max``, attributes.

This module applies these rules to data opened with Xarray, so it is masked in
the same way as when it is loaded with Iris. The rules follow
``netCDF4.Variable._toma``.

It also makes sure that decoded times get the same calendar as when they are
loaded with Iris, see :func:`set_default_calendar`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from netCDF4 import default_fillvals

if TYPE_CHECKING:
    from collections.abc import Collection

    import xarray as xr

logger = logging.getLogger(__name__)

RANGE_ATTRIBUTES = ("valid_range", "valid_min", "valid_max")
"""Attributes that define the range of valid values of a variable."""


def _cast_safely(name: str, value: Any, dtype: np.dtype) -> np.ndarray | None:  # noqa: ANN401
    """Cast an attribute value to ``dtype``, if this does not change it.

    The netCDF4 library ignores attributes that cannot be cast to the data
    type of the variable without changing their value.
    """
    original = np.asarray(value)
    try:
        cast = original.astype(dtype)
        is_safe = bool(
            ((original == cast) | (np.isnan(original) & np.isnan(cast))).all(),
        )
    except (TypeError, ValueError):
        is_safe = False
    if not is_safe:
        logger.debug(
            "Not using attribute %s = %r, because it cannot be cast safely "
            "to %s",
            name,
            value,
            dtype,
        )
        return None
    return cast


def _valid_range(
    attributes: dict[str, Any],
    dtype: np.dtype,
) -> tuple[np.ndarray | None, np.ndarray | None, list[str]]:
    """Get the valid range in the packed data type of a variable.

    Returns the minimum, the maximum and the names of the attributes used.
    """
    if "valid_range" in attributes:
        valid_range = _cast_safely(
            "valid_range",
            attributes["valid_range"],
            dtype,
        )
        if valid_range is not None and valid_range.size == 2:
            return valid_range[0], valid_range[1], ["valid_range"]
    valid_min = valid_max = None
    used = []
    if "valid_min" in attributes:
        valid_min = _cast_safely("valid_min", attributes["valid_min"], dtype)
        if valid_min is not None:
            used.append("valid_min")
    if "valid_max" in attributes:
        valid_max = _cast_safely("valid_max", attributes["valid_max"], dtype)
        if valid_max is not None:
            used.append("valid_max")
    return valid_min, valid_max, used


def _is_unsigned(encoding: dict[str, Any]) -> bool:
    """Check if signed packed data represents unsigned values."""
    return str(encoding.get("_Unsigned", "false")).lower() == "true"


def _decode(
    value: np.ndarray,
    encoding: dict[str, Any],
    dtype: np.dtype,
) -> np.ndarray:
    """Convert a value in the packed data type to the unpacked data type.

    This uses the same operations as Xarray, so the result is identical to
    the decoded data.
    """
    if _is_unsigned(encoding) and value.dtype.kind == "i":
        value = value.view(value.dtype.str.replace("i", "u"))
    result = np.asarray(value).astype(dtype, copy=True)
    if encoding.get("scale_factor") is not None:
        result *= encoding["scale_factor"]
    if encoding.get("add_offset") is not None:
        result += encoding["add_offset"]
    return result


def _fix_encoding(
    encoding: dict[str, Any],
    packed_dtype: np.dtype,
) -> dict[str, Any] | None:
    """Make the encoding work with the conversion to Iris cubes.

    The data is converted to Iris cubes with :mod:`ncdata`, which encodes it
    with Xarray. Iris then only masks values equal to ``_FillValue`` and does
    not apply ``_Unsigned``, so

    - ``_FillValue`` is set to ``missing_value`` if only the latter is
      present, and
    - unsigned data stored as signed integers with ``_Unsigned = "true"`` is
      encoded as unsigned integers instead.

    Returns the new encoding, or :obj:`None` if nothing needs to change.
    """
    changed = False
    encoding = dict(encoding)
    if _is_unsigned(encoding) and packed_dtype.kind == "i":
        unsigned = np.dtype(packed_dtype.str.replace("i", "u"))
        changed = True
        encoding["dtype"] = unsigned
        encoding.pop("_Unsigned")
        for key in ("_FillValue", "missing_value"):
            if key in encoding:
                encoding[key] = np.asarray(encoding[key], packed_dtype).view(
                    unsigned,
                )[()]
    if "_FillValue" not in encoding and "missing_value" in encoding:
        changed = True
        encoding["_FillValue"] = np.ravel(encoding["missing_value"])[0]
    return encoding if changed else None


def mask_like_netcdf4(
    dataset: xr.Dataset,
    no_fill: Collection[str] = (),
) -> None:
    """Mask the data in ``dataset`` in place, like the netCDF4 library does.

    Values that the netCDF4 library masks, but Xarray does not, are replaced
    by NaN. If the variable has no ``_FillValue`` in its encoding, the
    ``missing_value`` or the netCDF default fill value is added, so the values
    are masked again when the data is converted to Iris cubes. The range
    attributes that have been applied are removed, because they describe the
    packed data and are no longer needed.

    Index coordinates are not changed, because they become dimension
    coordinates in Iris, which cannot be masked.

    Parameters
    ----------
    dataset:
        Data opened with :func:`xarray.open_dataset` or
        :func:`xarray.open_mfdataset`, with the default ``mask_and_scale``.
    no_fill:
        Names of the variables in the file for which filling is turned off.
        For these variables, default fill values of byte types are not
        masked. By default, filling is assumed to be on for all variables.
    """
    for name, variable in list(dataset.variables.items()):
        encoding = variable.encoding
        packed_dtype = np.dtype(encoding.get("dtype", variable.dtype))
        if (
            name in dataset.indexes
            or variable.dtype.kind not in "fiu"
            or packed_dtype.kind not in "fiu"
        ):
            continue
        scale = encoding.get("scale_factor")
        reverse = scale is not None and scale < 0

        mask = None
        # The netCDF4 library compares unsigned data with the signed default
        # fill value, so default fill values are never masked for them.
        if (
            "_FillValue" not in encoding
            and not _is_unsigned(encoding)
            and (packed_dtype.itemsize > 1 or name not in no_fill)
        ):
            fill_value = np.array(
                default_fillvals[packed_dtype.str[1:]],
                packed_dtype,
            )
            mask = variable == _decode(fill_value, encoding, variable.dtype)

        valid_min, valid_max, used = _valid_range(variable.attrs, packed_dtype)
        for limit, below in ((valid_min, True), (valid_max, False)):
            if limit is None:
                continue
            decoded_limit = _decode(limit, encoding, variable.dtype)
            outside = (
                variable < decoded_limit
                if below != reverse
                else variable > decoded_limit
            )
            mask = outside if mask is None else mask | outside

        new_encoding = _fix_encoding(encoding, packed_dtype)
        if mask is None and new_encoding is None:
            continue
        if new_encoding is None:
            new_encoding = dict(encoding)
        if mask is not None and "_FillValue" not in new_encoding:
            dtype = np.dtype(new_encoding.get("dtype", packed_dtype))
            new_encoding["_FillValue"] = np.array(
                default_fillvals[dtype.str[1:]],
                dtype,
            )[()]
        new = (
            variable.copy(deep=False)
            if mask is None
            else variable.where(~mask)
        )
        new.attrs = {
            key: value
            for key, value in variable.attrs.items()
            if key not in used
        }
        new.encoding = new_encoding
        if name in dataset.coords:
            dataset.coords[name] = new
        else:
            dataset[name] = new


def set_default_calendar(dataset: xr.Dataset) -> None:
    """Set the calendar of decoded times without one to ``standard``.

    If a time variable has no ``calendar`` attribute, Iris uses the CF default
    calendar ``standard``. Xarray decodes such times to
    :class:`numpy.datetime64` and encodes them with the calendar
    ``proleptic_gregorian`` when the data is converted to Iris cubes. For
    dates after 1582 the two calendars are identical, but the cubes would
    have a different calendar than when the file is loaded with Iris.
    """
    for variable in dataset.variables.values():
        if (
            variable.dtype.kind == "M"
            and "units" in variable.encoding
            and "calendar" not in variable.encoding
        ):
            variable.encoding["calendar"] = "standard"
