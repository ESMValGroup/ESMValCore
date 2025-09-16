"""Auxiliary functions for :mod:`iris`."""

from __future__ import annotations

import contextlib
import warnings
from typing import TYPE_CHECKING, Any, Literal

import dask.array as da
import iris
import iris.cube
import iris.util
import ncdata
import ncdata.iris
import ncdata.iris_xarray
import ncdata.threadlock_sharing
import numpy as np
import xarray as xr
from cf_units import Unit, suppress_errors
from iris.cube import Cube
from iris.exceptions import CoordinateMultiDimError, CoordinateNotFoundError
from iris.warnings import IrisVagueMetadataWarning

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable, Sequence
    from pathlib import Path

    from iris.coords import Coord, DimCoord

    from esmvalcore.typing import NetCDFAttr

# Enable lock sharing between ncdata and iris/xarray
ncdata.threadlock_sharing.enable_lockshare(iris=True, xarray=True)


@contextlib.contextmanager
def ignore_iris_vague_metadata_warnings() -> Generator[None]:
    """Ignore specific warnings.

    This can be used as a context manager. See also
    https://scitools-iris.readthedocs.io/en/stable/generated/api/iris.warnings.html#iris.warnings.IrisVagueMetadataWarning.

    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=IrisVagueMetadataWarning,
            module="iris",
        )
        yield


def add_leading_dim_to_cube(cube: Cube, dim_coord: DimCoord) -> Cube:
    """Add new leading dimension to cube.

    An input cube with shape ``(x, ..., z)`` will be transformed to a cube with
    shape ``(w, x, ..., z)`` where ``w`` is the length of ``dim_coord``. Note
    that the data is broadcasted to the new shape.

    Parameters
    ----------
    cube:
        Input cube.
    dim_coord:
        Dimensional coordinate that is used to describe the new leading
        dimension. Needs to be 1D.

    Returns
    -------
    iris.cube.Cube
        Transformed input cube with new leading dimension.

    Raises
    ------
    iris.exceptions.CoordinateMultiDimError
        ``dim_coord`` is not 1D.

    """
    # Only 1D dim_coords are supported
    if dim_coord.ndim > 1:
        raise CoordinateMultiDimError(dim_coord)
    new_shape = (dim_coord.shape[0], *cube.shape)

    # Cache ancillary variables and cell measures (iris.util.new_axis drops
    # those) and determine corresponding dimensions in new cube
    ancillary_variables = []
    for ancillary_variable in cube.ancillary_variables():
        new_dims = tuple(
            d + 1 for d in cube.ancillary_variable_dims(ancillary_variable)
        )
        ancillary_variables.append((ancillary_variable, new_dims))
    cell_measures = []
    for cell_measure in cube.cell_measures():
        new_dims = tuple(d + 1 for d in cube.cell_measure_dims(cell_measure))
        cell_measures.append((cell_measure, new_dims))

    # Transform cube from shape (x, ..., z) to (1, x, ..., z)
    cube = iris.util.new_axis(cube)

    # Create new cube with shape (w, x, ..., z) where w is length of dim_coord
    # and already add ancillary variables and cell measures
    new_data = da.broadcast_to(cube.core_data(), new_shape)
    new_cube = Cube(
        new_data,
        ancillary_variables_and_dims=ancillary_variables,
        cell_measures_and_dims=cell_measures,
    )

    # Add metadata
    # Note: using cube.coord_dims() for determining the positions for the
    # coordinates of the new cube is correct here since cube has the shape (1,
    # x, ..., z) at this stage
    new_cube.metadata = cube.metadata
    new_cube.add_dim_coord(dim_coord, 0)
    for coord in cube.coords(dim_coords=True):
        new_cube.add_dim_coord(coord, cube.coord_dims(coord))
    for coord in cube.coords(dim_coords=False):
        new_cube.add_aux_coord(coord, cube.coord_dims(coord))

    return new_cube


def date2num(date, unit, dtype=np.float64):
    """Convert datetime object into numeric value with requested dtype.

    This is a custom version of :meth:`cf_units.Unit.date2num` that
    guarantees the correct dtype for the return value.

    Parameters
    ----------
    date : :class:`datetime.datetime` or :class:`cftime.datetime`
    unit : :class:`cf_units.Unit`
    dtype : a numpy dtype

    Returns
    -------
    :class:`numpy.ndarray` of type `dtype`
        The return value of ``unit.date2num`` with the requested dtype.
    """
    num = unit.date2num(date)
    try:
        return num.astype(dtype)
    except AttributeError:
        return dtype(num)


def merge_cube_attributes(  # noqa: C901
    cubes: Sequence[Cube],
    delimiter: str = " ",
) -> None:
    """Merge attributes of all given cubes in-place.

    After this operation, the attributes of all given cubes are equal. This is
    useful for operations that combine cubes, such as
    :meth:`iris.cube.CubeList.merge_cube` or
    :meth:`iris.cube.CubeList.concatenate_cube`.

    Note
    ----
    This function differs from :func:`iris.util.equalise_attributes` in this
    respect that it does not delete attributes that are not identical but
    rather concatenates them (sorted) using the given ``delimiter``. E.g., the
    attributes ``exp: historical`` and ``exp: ssp585`` end up as ``exp:
    historical ssp585`` using the default ``delimiter = ' '``.

    Parameters
    ----------
    cubes:
        Input cubes whose attributes will be modified in-place.
    delimiter:
        Delimiter that is used to concatenate non-identical attributes.

    """
    if len(cubes) <= 1:
        return

    # Step 1: collect all attribute values in a list
    attributes: dict[str, list[NetCDFAttr]] = {}
    for cube in cubes:
        for attr, val in cube.attributes.items():
            attributes.setdefault(attr, [])
            attributes[attr].append(val)

    # Step 2: use the first cube in which an attribute occurs to decide if an
    # attribute is global or local.
    final_attributes = iris.cube.CubeAttrsDict()
    for cube in cubes:
        for attr, value in cube.attributes.locals.items():
            if attr not in final_attributes:
                final_attributes.locals[attr] = value
        for attr, value in cube.attributes.globals.items():
            if attr not in final_attributes:
                final_attributes.globals[attr] = value

    # Step 3: if values are not equal, first convert them to strings (so that
    # set() can be used); then extract unique elements from this list, sort it,
    # and use the delimiter to join all elements to a single string.
    for attr, vals in attributes.items():
        set_of_str = sorted({str(v) for v in vals})
        if len(set_of_str) == 1:
            final_attributes[attr] = vals[0]
        else:
            final_attributes[attr] = delimiter.join(set_of_str)

    # Step 4: modify the cubes in-place
    for cube in cubes:
        cube.attributes = final_attributes


def _rechunk(
    array: da.core.Array,
    complete_dims: list[int],
    remaining_dims: int | Literal["auto"],
) -> da.core.Array:
    """Rechunk a given array so that it is not chunked along given dims."""
    new_chunks: list[str | int] = [remaining_dims] * array.ndim
    for dim in complete_dims:
        new_chunks[dim] = -1
    return array.rechunk(new_chunks)


def _rechunk_dim_metadata(
    cube: Cube,
    complete_dims: Iterable[int],
    remaining_dims: int | Literal["auto"] = "auto",
) -> None:
    """Rechunk dimensional metadata of a cube (in-place)."""
    # Non-dimensional coords that span complete_dims
    # Note: dimensional coords are always realized (i.e., numpy arrays), so no
    # chunking is necessary
    for coord in cube.coords(dim_coords=False):
        dims = cube.coord_dims(coord)
        complete_dims_ = [dims.index(d) for d in complete_dims if d in dims]
        if complete_dims_:
            if coord.has_lazy_points():
                coord.points = _rechunk(
                    coord.lazy_points(),
                    complete_dims_,
                    remaining_dims,
                )
            if coord.has_bounds() and coord.has_lazy_bounds():
                coord.bounds = _rechunk(
                    coord.lazy_bounds(),
                    complete_dims_,
                    remaining_dims,
                )

    # Rechunk cell measures that span complete_dims
    for measure in cube.cell_measures():
        dims = cube.cell_measure_dims(measure)
        complete_dims_ = [dims.index(d) for d in complete_dims if d in dims]
        if complete_dims_ and measure.has_lazy_data():
            measure.data = _rechunk(
                measure.lazy_data(),
                complete_dims_,
                remaining_dims,
            )

    # Rechunk ancillary variables that span complete_dims
    for anc_var in cube.ancillary_variables():
        dims = cube.ancillary_variable_dims(anc_var)
        complete_dims_ = [dims.index(d) for d in complete_dims if d in dims]
        if complete_dims_ and anc_var.has_lazy_data():
            anc_var.data = _rechunk(
                anc_var.lazy_data(),
                complete_dims_,
                remaining_dims,
            )


def rechunk_cube(
    cube: Cube,
    complete_coords: Iterable[Coord | str],
    remaining_dims: int | Literal["auto"] = "auto",
) -> Cube:
    """Rechunk cube so that it is not chunked along given dimensions.

    This will rechunk the cube's data, but also all non-dimensional
    coordinates, cell measures, and ancillary variables that span at least one
    of the given dimensions.

    Note
    ----
    This will only rechunk `dask` arrays. `numpy` arrays are not changed.

    Parameters
    ----------
    cube:
        Input cube.
    complete_coords:
        (Names of) coordinates along which the output cube should not be
        chunked.
    remaining_dims:
        Chunksize of the remaining dimensions.

    Returns
    -------
    iris.cube.Cube
        Rechunked cube. This will always be a copy of the input cube.

    """
    cube = cube.copy()  # do not modify input cube

    complete_dims = []
    for coord in complete_coords:
        complete_dims.extend(cube.coord_dims(cube.coord(coord)))
    complete_dims = list(set(complete_dims))

    # Rechunk data
    if cube.has_lazy_data():
        cube.data = _rechunk(cube.lazy_data(), complete_dims, remaining_dims)

    # Rechunk dimensional metadata
    _rechunk_dim_metadata(cube, complete_dims, remaining_dims=remaining_dims)

    return cube


def has_regular_grid(cube: Cube) -> bool:
    """Check if a cube has a regular grid.

    "Regular" refers to a rectilinear grid with 1D latitude and 1D longitude
    coordinates orthogonal to each other.

    Parameters
    ----------
    cube:
        Cube to be checked.

    Returns
    -------
    bool
        ``True`` if input cube has a regular grid, else ``False``.

    """
    try:
        lat = cube.coord("latitude")
        lon = cube.coord("longitude")
    except CoordinateNotFoundError:
        return False
    if lat.ndim != 1 or lon.ndim != 1:
        return False
    return cube.coord_dims(lat) != cube.coord_dims(lon)


def has_irregular_grid(cube: Cube) -> bool:
    """Check if a cube has an irregular grid.

    "Irregular" refers to a general curvilinear grid with 2D latitude and 2D
    longitude coordinates with common dimensions.

    Parameters
    ----------
    cube:
        Cube to be checked.

    Returns
    -------
    bool
        ``True`` if input cube has an irregular grid, else ``False``.

    """
    try:
        lat = cube.coord("latitude")
        lon = cube.coord("longitude")
    except CoordinateNotFoundError:
        return False
    return bool(lat.ndim == 2 and lon.ndim == 2)


def has_unstructured_grid(cube: Cube) -> bool:
    """Check if a cube has an unstructured grid.

    "Unstructured" refers to a grid with 1D latitude and 1D longitude
    coordinates with common dimensions (i.e., a simple list of points).

    Parameters
    ----------
    cube:
        Cube to be checked.

    Returns
    -------
    bool
        ``True`` if input cube has an unstructured grid, else ``False``.

    """
    try:
        lat = cube.coord("latitude")
        lon = cube.coord("longitude")
    except CoordinateNotFoundError:
        return False
    if lat.ndim != 1 or lon.ndim != 1:
        return False
    return cube.coord_dims(lat) == cube.coord_dims(lon)


# List containing special cases for unit conversion. Each list item is another
# list. Each of these sublists defines one special conversion. Each element in
# the sublists is a tuple (standard_name, units). Note: All units for a single
# special case need to be "physically identical", e.g., 1 kg m-2 s-1 "equals" 1
# mm s-1 for precipitation
_SPECIAL_UNIT_CONVERSIONS: list[list[tuple[str | None, str]]] = [
    [
        ("precipitation_flux", "kg m-2 s-1"),
        ("lwe_precipitation_rate", "mm s-1"),
    ],
    [
        ("water_evaporation_flux", "kg m-2 s-1"),
        ("lwe_water_evaporation_rate", "mm s-1"),
    ],
    [
        ("water_potential_evaporation_flux", "kg m-2 s-1"),
        (None, "mm s-1"),  # no standard_name for potential evaporation rate
    ],
    [
        ("equivalent_thickness_at_stp_of_atmosphere_ozone_content", "m"),
        ("equivalent_thickness_at_stp_of_atmosphere_ozone_content", "1e5 DU"),
    ],
    [
        ("surface_air_pressure", "Pa"),
        ("atmosphere_mass_of_air_per_unit_area", "1/9.80665 kg m-2"),
    ],
]


def _try_special_unit_conversions(cube: Cube, units: str | Unit) -> bool:
    """Try special unit conversion (in-place).

    Parameters
    ----------
    cube:
        Input cube (modified in place).
    units:
        New units

    Returns
    -------
    bool
        ``True`` if special unit conversion was successful, ``False`` if not.

    """
    for special_case in _SPECIAL_UNIT_CONVERSIONS:
        for std_name, special_units in special_case:
            # Special unit conversion only works if all of the following
            # criteria are met:
            # - the cube's standard_name is one of the supported
            #   standard_names
            # - the cube's units are convertible to the ones defined for
            #   that given standard_name
            # - the desired target units are convertible to the units of
            #   one of the other standard_names in that special case

            # Step 1: find suitable source name and units
            if cube.standard_name == std_name and cube.units.is_convertible(
                special_units,
            ):
                for target_std_name, target_units in special_case:
                    if target_units == special_units:
                        continue

                    # Step 2: find suitable target name and units
                    if Unit(units).is_convertible(target_units):
                        cube.standard_name = target_std_name

                        # In order to avoid two calls to cube.convert_units,
                        # determine the conversion factor between the cube's
                        # units and the source units first and simply add this
                        # factor to the target units (remember that the source
                        # units and the target units should be "physically
                        # identical").
                        factor = cube.units.convert(1.0, special_units)
                        cube.units = f"{factor} {target_units}"
                        cube.convert_units(units)
                        return True

    # If no special case has been detected, return False
    return False


def safe_convert_units(cube: Cube, units: str | Unit) -> Cube:
    """Safe unit conversion (change of `standard_name` not allowed; in-place).

    This is a safe version of :func:`esmvalcore.preprocessor.convert_units`
    that will raise an error if the input cube's
    :attr:`~iris.cube.Cube.standard_name` has been changed.

    Parameters
    ----------
    cube:
        Input cube (modified in place).
    units:
        New units.

    Returns
    -------
    iris.cube.Cube
        Converted cube. Just returned for convenience; input cube is modified
        in place.

    Raises
    ------
    iris.exceptions.UnitConversionError
        Old units are unknown.
    ValueError
        Old units are not convertible to new units or unit conversion required
        change of `standard_name`.

    """
    old_units = cube.units
    old_standard_name = cube.standard_name

    try:
        cube.convert_units(units)
    except ValueError:
        if not _try_special_unit_conversions(cube, units):
            raise

    if cube.standard_name != old_standard_name:
        msg = (
            f"Cannot safely convert units from '{old_units}' to '{units}'; "
            f"standard_name changed from '{old_standard_name}' to "
            f"'{cube.standard_name}'"
        )
        raise ValueError(
            msg,
        )
    return cube


@contextlib.contextmanager
def ignore_warnings_context(
    warnings_to_ignore: list[dict[str, Any]] | None = None,
) -> Generator[None]:
    """Ignore warnings (context manager).

    Parameters
    ----------
    warnings_to_ignore:
        Additional warnings to ignore (by default, Iris warnings about missing
        CF-netCDF measure variables and invalid units are ignored).

    """
    if warnings_to_ignore is None:
        warnings_to_ignore = []

    default_warnings_to_ignore: list[dict[str, Any]] = [
        {
            "message": "Missing CF-netCDF measure variable .*",
            "category": UserWarning,
            "module": "iris",
        },
        {
            "message": "Ignoring invalid units .* on netCDF variable .*",
            "category": UserWarning,
            "module": "iris",
        },
    ]

    with contextlib.ExitStack() as stack:
        # Regular warnings
        stack.enter_context(warnings.catch_warnings())
        for warning_kwargs in warnings_to_ignore + default_warnings_to_ignore:
            warning_kwargs.setdefault("action", "ignore")
            warnings.filterwarnings(**warning_kwargs)

        # Suppress UDUNITS-2 error messages that cannot be ignored with
        # warnings.filterwarnings
        # (see https://github.com/SciTools/cf-units/issues/240)
        stack.enter_context(suppress_errors())

        yield


def _get_attribute(
    data: ncdata.NcData | ncdata.NcVariable | xr.Dataset | xr.DataArray,
    attribute_name: str,
) -> Any:
    """Get attribute from an ncdata or xarray object."""
    if isinstance(data, ncdata.NcData | ncdata.NcVariable):
        attribute = data.attributes[attribute_name].value
    else:  # xr.Dataset | xr.DataArray
        attribute = data.attrs[attribute_name]
    return attribute


def dataset_to_iris(
    dataset: xr.Dataset | ncdata.NcData,
    filepath: str | Path | None = None,
    ignore_warnings: list[dict[str, Any]] | None = None,
) -> iris.cube.CubeList:
    """Convert dataset to :class:`~iris.cube.CubeList`.

    Parameters
    ----------
    dataset:
        The dataset object to convert.
    filepath:
        The path that the dataset was loaded from.
    ignore_warnings:
        Keyword arguments passed to :func:`warnings.filterwarnings` used to
        ignore warnings during data loading. Each list element corresponds
        to one call to :func:`warnings.filterwarnings`.

    Returns
    -------
    iris.cube.CubeList
        :class:`~iris.cube.CubeList` containing the requested cubes.

    Raises
    ------
    TypeError
        Invalid type for ``dataset`` given.

    """
    if isinstance(dataset, xr.Dataset):
        conversion_func = ncdata.iris_xarray.cubes_from_xarray
        ds_coords = dataset.coords
    elif isinstance(dataset, ncdata.NcData):
        conversion_func = ncdata.iris.to_iris
        ds_coords = dataset.variables
    else:
        msg = (
            f"Expected type ncdata.NcData or xr.Dataset for dataset, got "
            f"type {type(dataset)}"
        )
        raise TypeError(
            msg,
        )

    with ignore_warnings_context(ignore_warnings):
        cubes = conversion_func(dataset)

    # Restore the lat/lon coordinate units that iris changes to degrees
    for coord_name in ["latitude", "longitude"]:
        for cube in cubes:
            try:
                coord = cube.coord(coord_name)
            except iris.exceptions.CoordinateNotFoundError:
                pass
            else:
                if coord.var_name in ds_coords:
                    ds_coord = ds_coords[coord.var_name]
                    coord.units = _get_attribute(ds_coord, "units")

            # If possible, add the source file as an attribute to support
            # grouping by file when calling fix_metadata.
            if filepath is not None:
                cube.attributes.globals["source_file"] = str(filepath)

    return cubes
