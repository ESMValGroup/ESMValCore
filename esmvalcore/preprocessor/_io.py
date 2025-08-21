"""Functions for loading and saving cubes."""

from __future__ import annotations

import copy
import logging
import os
import warnings
from itertools import groupby
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import fsspec
import iris
import ncdata
import xarray as xr
import yaml
from iris.cube import Cube, CubeList

from esmvalcore._task import write_ncl_settings
from esmvalcore.exceptions import ESMValCoreLoadWarning
from esmvalcore.iris_helpers import (
    dataset_to_iris,
    ignore_warnings_context,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dask.delayed import Delayed
    from iris.fileformats.cf import CFVariable

logger = logging.getLogger(__name__)

GLOBAL_FILL_VALUE = 1e20

DATASET_KEYS = {
    "mip",
}
VARIABLE_KEYS = {
    "reference_dataset",
    "alternative_dataset",
}
GRIB_FORMATS = (".grib2", ".grib", ".grb2", ".grb", ".gb2", ".gb")


def _get_attr_from_field_coord(
    ncfield: CFVariable,
    coord_name: str | None,
    attr: str,
) -> Any:
    """Get attribute from netCDF field coordinate."""
    if coord_name is not None:
        attrs = ncfield.cf_group[coord_name].cf_attrs()
        attr_val = [value for (key, value) in attrs if key == attr]
        if attr_val:
            return attr_val[0]
    return None


def _restore_lat_lon_units(
    cube: Cube,
    field: CFVariable,
    filename: str,  # noqa: ARG001
) -> None:  # pylint: disable=unused-argument
    """Use this callback to restore the original lat/lon units."""
    # Iris chooses to change longitude and latitude units to degrees
    # regardless of value in file, so reinstating file value
    for coord in cube.coords():
        if coord.standard_name in ["longitude", "latitude"]:
            units = _get_attr_from_field_coord(field, coord.var_name, "units")
            if units is not None:
                coord.units = units


def load(
    file: str | Path | Cube | CubeList | xr.Dataset | ncdata.NcData,
    ignore_warnings: list[dict[str, Any]] | None = None,
    backend_kwargs: dict[str, Any] | None = None,
) -> CubeList:
    """Load Iris cubes.

    Parameters
    ----------
    file:
        File to be loaded. If ``file`` is already a loaded dataset, return it
        as a :class:`~iris.cube.CubeList`.
        File as ``Path`` object could be a Zarr store.
    ignore_warnings:
        Keyword arguments passed to :func:`warnings.filterwarnings` used to
        ignore warnings issued by :func:`iris.load_raw`. Each list element
        corresponds to one call to :func:`warnings.filterwarnings`.
    backend_kwargs:
        Dict to hold info needed by storage backend e.g. to access
        a PRIVATE S3 bucket containing object stores (e.g. netCDF4 files);
        needed by ``fsspec`` and its extensions e.g. ``s3fs``, so
        most of the times this will include ``storage_options``. Note that Zarr
        files are opened via ``http`` extension of ``fsspec``, so no need
        for ``storage_options`` in that case (ie anon/anon). Currently only used
        in Zarr file opening.

    Returns
    -------
    iris.cube.CubeList
        Loaded cubes.

    Raises
    ------
    ValueError
        Cubes are empty.
    TypeError
        Invalid type for ``file``.

    """
    if isinstance(file, (str, Path)):
        extension = (
            file.suffix
            if isinstance(file, Path)
            else os.path.splitext(file)[1]
        )
        if "zarr" not in extension:
            cubes = _load_from_file(file, ignore_warnings=ignore_warnings)
        else:
            cubes = _load_zarr(
                file,
                ignore_warnings=ignore_warnings,
                backend_kwargs=backend_kwargs,
            )
    elif isinstance(file, Cube):
        cubes = CubeList([file])
    elif isinstance(file, CubeList):
        cubes = file
    elif isinstance(file, (xr.Dataset, ncdata.NcData)):
        cubes = dataset_to_iris(file, ignore_warnings=ignore_warnings)
    else:
        msg = (
            f"Expected type str, pathlib.Path, iris.cube.Cube, "
            f"iris.cube.CubeList, xarray.Dataset, or ncdata.NcData for file, "
            f"got type {type(file)}"
        )
        raise TypeError(msg)

    if not cubes:
        msg = f"{file} does not contain any data"
        raise ValueError(msg)

    for cube in cubes:
        if "source_file" not in cube.attributes:
            warn_msg = (
                f"Cube {cube.summary(shorten=True)} loaded from\n{file}\ndoes "
                f"not contain attribute 'source_file' that points to original "
                f"file path, please make sure to add it prior to loading "
                f"(preferably during the preprocessing step 'fix_file')"
            )
            warnings.warn(warn_msg, ESMValCoreLoadWarning, stacklevel=2)

    return cubes


def _load_zarr(
    file: str | Path | Cube | CubeList | xr.Dataset | ncdata.NcData,
    ignore_warnings: list[dict[str, Any]] | None = None,
    backend_kwargs: dict[str, Any] | None = None,
) -> CubeList:
    # note on ``chunks`` kwarg to ``xr.open_dataset()``
    # docs.xarray.dev/en/stable/generated/xarray.open_dataset.html
    # this is very important because with ``chunks=None`` (default)
    # data will be realized as Numpy arrays and transferred in memory;
    # ``chunks={}`` loads the data with dask using the engine preferred
    # chunk size, generally identical to the formats chunk size. If not
    # available, a single chunk for all arrays; testing shows this is the
    # "best guess" compromise for typically CMIP-like chunked data.
    # see https://github.com/pydata/xarray/issues/10612 and
    # https://github.com/pp-mo/ncdata/issues/139

    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    open_kwargs = {
        "consolidated": False,
        "decode_times": time_coder,
        "engine": "zarr",
        "chunks": {},
        "backend_kwargs": backend_kwargs,
    }

    # Case 1: Zarr store is on remote object store
    # file's URI will always be either http or https
    if urlparse(str(file)).scheme in ["http", "https"]:
        # basic test that opens the Zarr/.zmetadata file for Zarr2
        # or Zarr/zarr.json for Zarr3
        fs = fsspec.filesystem("http")
        valid_zarr = True
        try:
            fs.open(str(file) + "/zarr.json", "rb")  # Zarr3
        except Exception:  # noqa: BLE001
            try:
                fs.open(str(file) + "/.zmetadata", "rb")  # Zarr2
            except Exception:  # noqa: BLE001
                valid_zarr = False
        # we don't want to catch any specific aiohttp/fsspec exception
        # bottom line is that that file has issues, so raise
        if not valid_zarr:
            msg = (
                f"File '{file}' can not be opened as Zarr file at the moment."
            )
            raise ValueError(msg)

        open_kwargs["consolidated"] = True
        zarr_xr = xr.open_dataset(file, **open_kwargs)
    # Case 2: Zarr store is local to the file system
    else:
        zarr_xr = xr.open_dataset(file, **open_kwargs)

    # avoid possible
    # ValueError: Object has inconsistent chunks along dimension time.
    # This can be fixed by calling unify_chunks().
    # when trying to access the ``chunks`` store
    zarr_xr = zarr_xr.unify_chunks()

    return dataset_to_iris(zarr_xr, ignore_warnings=ignore_warnings)


def _load_from_file(
    file: str | Path,
    ignore_warnings: list[dict[str, Any]] | None = None,
) -> CubeList:
    """Load data from file."""
    file = Path(file)
    logger.debug("Loading:\n%s", file)

    with ignore_warnings_context(ignore_warnings):
        # GRIB files need to be loaded with iris.load, otherwise we will
        # get separate (lat, lon) slices for each time step, pressure
        # level, etc.
        if file.suffix in GRIB_FORMATS:
            cubes = iris.load(file, callback=_restore_lat_lon_units)
        else:
            cubes = iris.load_raw(file, callback=_restore_lat_lon_units)
    logger.debug("Done with loading %s", file)

    for cube in cubes:
        cube.attributes.globals["source_file"] = str(file)

    return cubes


def save(  # noqa: C901
    cubes: Sequence[Cube],
    filename: Path | str,
    optimize_access: str = "",
    compress: bool = False,
    alias: str = "",
    compute: bool = True,
    **kwargs,
) -> Delayed | None:
    """Save iris cubes to file.

    Parameters
    ----------
    cubes:
        Data cubes to be saved

    filename:
        Name of target file

    optimize_access:
        Set internal NetCDF chunking to favour a reading scheme

        Values can be map or timeseries, which improve performance when
        reading the file one map or time series at a time.
        Users can also provide a coordinate or a list of coordinates. In that
        case the better performance will be avhieved by loading all the values
        in that coordinate at a time

    compress:
        Use NetCDF internal compression.

    alias:
        Var name to use when saving instead of the one in the cube.

    compute : bool, default=True
        Default is ``True``, meaning complete the file immediately, and return
        ``None``.

        When ``False``, create the output file but don't write any lazy array
        content to its variables, such as lazy cube data or aux-coord points
        and bounds.  Instead return a :class:`dask.delayed.Delayed` which, when
        computed, will stream all the lazy content via :meth:`dask.store`, to
        complete the file.  Several such data saves can be performed in
        parallel, by passing a list of them into a :func:`dask.compute` call.

    **kwargs:
        See :func:`iris.fileformats.netcdf.saver.save` for additional
        keyword arguments.

    Returns
    -------
    :class:`dask.delayed.Delayed` or :obj:`None`
        A delayed object that can be used to save the data in the cube.

    Raises
    ------
    ValueError
        cubes is empty.
    """
    if not cubes:
        msg = f"Cannot save empty cubes '{cubes}'"
        raise ValueError(msg)

    if Path(filename).suffix.lower() == ".nc":
        kwargs["compute"] = compute

    # Rename some arguments
    kwargs["target"] = filename
    kwargs["zlib"] = compress

    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    if os.path.exists(filename) and all(
        cube.has_lazy_data() for cube in cubes
    ):
        logger.debug(
            "Not saving cubes %s to %s to avoid data loss. "
            "The cube is probably unchanged.",
            cubes,
            filename,
        )
        return None

    for cube in cubes:
        logger.debug(
            "Saving cube:\n%s\nwith %s data to %s",
            cube,
            "lazy" if cube.has_lazy_data() else "realized",
            filename,
        )
    if optimize_access:
        cube = cubes[0]
        if optimize_access == "map":
            dims = set(
                cube.coord_dims("latitude") + cube.coord_dims("longitude"),
            )
        elif optimize_access == "timeseries":
            dims = set(cube.coord_dims("time"))
        else:
            dims = {
                dim
                for coord_name in optimize_access.split(" ")
                for dim in cube.coord_dims(coord_name)
            }

        kwargs["chunksizes"] = tuple(
            length if index in dims else 1
            for index, length in enumerate(cube.shape)
        )

    kwargs["fill_value"] = GLOBAL_FILL_VALUE
    if alias:
        for cube in cubes:
            logger.debug(
                "Changing var_name from %s to %s",
                cube.var_name,
                alias,
            )
            cube.var_name = alias

    # Ignore some warnings when saving
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                ".* is being added as CF data variable attribute, but .* "
                "should only be a CF global attribute"
            ),
            category=UserWarning,
            module="iris",
        )
        return iris.save(cubes, **kwargs)


def _get_debug_filename(filename, step):
    """Get a filename for debugging the preprocessor."""
    dirname = os.path.splitext(filename)[0]
    if os.path.exists(dirname) and os.listdir(dirname):
        num = int(sorted(os.listdir(dirname)).pop()[:2]) + 1
    else:
        num = 0
    return os.path.join(dirname, f"{num:02}_{step}.nc")


def _sort_products(products):
    """Sort preprocessor output files by their order in the recipe."""
    return sorted(
        products,
        key=lambda p: (
            p.attributes.get("recipe_dataset_index", 1e6),
            p.attributes.get("dataset", ""),
        ),
    )


def write_metadata(products, write_ncl=False):
    """Write product metadata to file."""
    output_files = []
    for output_dir, prods in groupby(
        products,
        lambda p: os.path.dirname(p.filename),
    ):
        sorted_products = _sort_products(prods)
        metadata = {}
        for product in sorted_products:
            if isinstance(product.attributes.get("exp"), (list, tuple)):
                product.attributes = dict(product.attributes)
                product.attributes["exp"] = "-".join(product.attributes["exp"])
            if "original_short_name" in product.attributes:
                del product.attributes["original_short_name"]
            metadata[product.filename] = product.attributes

        output_filename = os.path.join(output_dir, "metadata.yml")
        output_files.append(output_filename)
        with open(output_filename, "w", encoding="utf-8") as file:
            yaml.safe_dump(metadata, file)
        if write_ncl:
            output_files.append(_write_ncl_metadata(output_dir, metadata))

    return output_files


def _write_ncl_metadata(output_dir, metadata):
    """Write NCL metadata files to output_dir."""
    variables = [copy.deepcopy(v) for v in metadata.values()]

    info = {"input_file_info": variables}

    # Split input_file_info into dataset and variable properties
    # dataset keys and keys with non-identical values will be stored
    # in dataset_info, the rest in variable_info
    variable_info = {}
    info["variable_info"] = [variable_info]
    info["dataset_info"] = []
    for variable in variables:
        dataset_info = {}
        info["dataset_info"].append(dataset_info)
        for key in variable:
            dataset_specific = any(
                variable[key] != var.get(key, object()) for var in variables
            )
            if (
                dataset_specific or key in DATASET_KEYS
            ) and key not in VARIABLE_KEYS:
                dataset_info[key] = variable[key]
            else:
                variable_info[key] = variable[key]

    filename = os.path.join(
        output_dir,
        variable_info["short_name"] + "_info.ncl",
    )
    write_ncl_settings(info, filename)

    return filename
