"""Functions for loading and saving cubes."""
from __future__ import annotations

import copy
import logging
import os
from itertools import groupby
from pathlib import Path
from typing import NamedTuple, Optional
from warnings import catch_warnings, filterwarnings

import cftime
import iris
import iris.aux_factory
import iris.exceptions
import numpy as np
import yaml
from cf_units import suppress_errors
from iris.cube import CubeList

from esmvalcore.cmor.check import CheckLevels
from esmvalcore.iris_helpers import merge_cube_attributes

from .._task import write_ncl_settings

logger = logging.getLogger(__name__)

GLOBAL_FILL_VALUE = 1e+20

DATASET_KEYS = {
    'mip',
}
VARIABLE_KEYS = {
    'reference_dataset',
    'alternative_dataset',
}

iris.FUTURE.save_split_attrs = True


def _get_attr_from_field_coord(ncfield, coord_name, attr):
    if coord_name is not None:
        attrs = ncfield.cf_group[coord_name].cf_attrs()
        attr_val = [value for (key, value) in attrs if key == attr]
        if attr_val:
            return attr_val[0]
    return None


def _load_callback(raw_cube, field, _):
    """Use this callback to fix anything Iris tries to break."""
    # Remove attributes that cause issues with merging and concatenation
    _delete_attributes(raw_cube,
                       ('creation_date', 'tracking_id', 'history', 'comment'))
    for coord in raw_cube.coords():
        # Iris chooses to change longitude and latitude units to degrees
        # regardless of value in file, so reinstating file value
        if coord.standard_name in ['longitude', 'latitude']:
            units = _get_attr_from_field_coord(field, coord.var_name, 'units')
            if units is not None:
                coord.units = units
        # CMOR sometimes adds a history to the coordinates.
        _delete_attributes(coord, ('history', ))


def _delete_attributes(iris_object, atts):
    for att in atts:
        if att in iris_object.attributes:
            del iris_object.attributes[att]


def load(
    file: str | Path,
    ignore_warnings: Optional[list[dict]] = None,
) -> CubeList:
    """Load iris cubes from string or Path objects.

    Parameters
    ----------
    file:
        File to be loaded. Could be string or POSIX Path object.
    ignore_warnings:
        Keyword arguments passed to :func:`warnings.filterwarnings` used to
        ignore warnings issued by :func:`iris.load_raw`. Each list element
        corresponds to one call to :func:`warnings.filterwarnings`.

    Returns
    -------
    iris.cube.CubeList
        Loaded cubes.

    Raises
    ------
    ValueError
        Cubes are empty.
    """
    file = Path(file)
    logger.debug("Loading:\n%s", file)

    if ignore_warnings is None:
        ignore_warnings = []

    # Avoid duplication of ignored warnings when load() is called more often
    # than once
    ignore_warnings = list(ignore_warnings)

    # Default warnings ignored for every dataset
    ignore_warnings.append({
        'message': "Missing CF-netCDF measure variable .*",
        'category': UserWarning,
        'module': 'iris',
    })
    ignore_warnings.append({
        'message': "Ignoring netCDF variable '.*' invalid units '.*'",
        'category': UserWarning,
        'module': 'iris',
    })  # iris < 3.8
    ignore_warnings.append({
        'message': "Ignoring invalid units .* on netCDF variable .*",
        'category': UserWarning,
        'module': 'iris',
    })  # iris >= 3.8

    # Filter warnings
    with catch_warnings():
        for warning_kwargs in ignore_warnings:
            warning_kwargs.setdefault('action', 'ignore')
            filterwarnings(**warning_kwargs)
        # Suppress UDUNITS-2 error messages that cannot be ignored with
        # warnings.filterwarnings
        # (see https://github.com/SciTools/cf-units/issues/240)
        with suppress_errors():
            raw_cubes = iris.load_raw(file, callback=_load_callback)
    logger.debug("Done with loading %s", file)

    if not raw_cubes:
        raise ValueError(f'Can not load cubes from {file}')

    for cube in raw_cubes:
        cube.attributes['source_file'] = str(file)

    return raw_cubes


def _concatenate_cubes(cubes, check_level):
    """Concatenate cubes according to the check_level."""
    kwargs = {
        'check_aux_coords': True,
        'check_cell_measures': True,
        'check_ancils': True,
        'check_derived_coords': True
    }

    if check_level > CheckLevels.DEFAULT:
        kwargs = dict.fromkeys(kwargs, False)
        logger.debug(
            'Concatenation will be performed without checking '
            'auxiliary coordinates, cell measures, ancillaries '
            'and derived coordinates present in the cubes.', )

    concatenated = iris.cube.CubeList(cubes).concatenate(**kwargs)

    return concatenated


class _TimesHelper:

    def __init__(self, time):
        self.times = time.core_points()
        self.units = str(time.units)

    def __getattr__(self, name):
        return getattr(self.times, name)

    def __len__(self):
        return len(self.times)

    def __getitem__(self, key):
        return self.times[key]


def _check_time_overlaps(cubes: iris.cube.CubeList) -> iris.cube.CubeList:
    """Handle time overlaps.

    Parameters
    ----------
    cubes : iris.cube.CubeList
        A list of cubes belonging to a single timeseries,
        ordered by starting point with possible overlaps.

    Returns
    -------
    iris.cube.CubeList
        A list of cubes belonging to a single timeseries,
        ordered by starting point with no overlaps.
    """
    if len(cubes) < 2:
        return cubes

    class _TrackedCube(NamedTuple):
        cube: iris.cube.Cube
        times: iris.coords.DimCoord
        start: float
        end: float

        @classmethod
        def from_cube(cls, cube):
            """Construct tracked cube."""
            times = cube.coord("time")
            start, end = times.core_points()[[0, -1]]
            return cls(cube, times, start, end)

    new_cubes = iris.cube.CubeList()
    current_cube = _TrackedCube.from_cube(cubes[0])
    for new_cube in map(_TrackedCube.from_cube, cubes[1:]):
        if new_cube.start > current_cube.end:
            # no overlap, use current cube and start again from new cube
            logger.debug("Using %s", current_cube.cube)
            new_cubes.append(current_cube.cube)
            current_cube = new_cube
            continue
        # overlap
        if current_cube.end > new_cube.end:
            # current cube ends after new one, just forget new cube
            logger.debug(
                "Discarding %s because the time range "
                "is already covered by %s", new_cube.cube, current_cube.cube)
            continue
        if new_cube.start == current_cube.start:
            # new cube completely covers current one
            # forget current cube
            current_cube = new_cube
            logger.debug(
                "Discarding %s because the time range is covered by %s",
                current_cube.cube, new_cube.cube)
            continue
        # new cube ends after current one,
        # use all of new cube, and shorten current cube to
        # eliminate overlap with new cube
        cut_index = cftime.time2index(
            new_cube.start,
            _TimesHelper(current_cube.times),
            current_cube.times.units.calendar,
            select="before",
        ) + 1
        logger.debug("Using %s shortened to %s due to overlap",
                     current_cube.cube,
                     current_cube.times.cell(cut_index).point)
        new_cubes.append(current_cube.cube[:cut_index])
        current_cube = new_cube

    logger.debug("Using %s", current_cube.cube)
    new_cubes.append(current_cube.cube)

    return new_cubes


def _fix_calendars(cubes):
    """Check and homogenise calendars, if possible."""
    calendars = [cube.coord('time').units.calendar for cube in cubes]
    unique_calendars = np.unique(calendars)

    calendar_ocurrences = np.array(
        [calendars.count(calendar) for calendar in unique_calendars])
    calendar_index = int(
        np.argwhere(calendar_ocurrences == calendar_ocurrences.max()))

    for cube in cubes:
        time_coord = cube.coord('time')
        old_calendar = time_coord.units.calendar
        if old_calendar != unique_calendars[calendar_index]:
            new_unit = time_coord.units.change_calendar(
                unique_calendars[calendar_index])
            time_coord.units = new_unit


def _get_concatenation_error(cubes):
    """Raise an error for concatenation."""
    # Concatenation not successful -> retrieve exact error message
    try:
        iris.cube.CubeList(cubes).concatenate_cube()
    except iris.exceptions.ConcatenateError as exc:
        msg = str(exc)
    logger.error('Can not concatenate cubes into a single one: %s', msg)
    logger.error('Resulting cubes:')
    for cube in cubes:
        logger.error(cube)
        time = cube.coord("time")
        logger.error('From %s to %s', time.cell(0), time.cell(-1))

    raise ValueError(f'Can not concatenate cubes: {msg}')


def _sort_cubes_by_time(cubes):
    """Sort CubeList by time coordinate."""
    try:
        cubes = sorted(cubes, key=lambda c: c.coord("time").cell(0).point)
    except iris.exceptions.CoordinateNotFoundError as exc:
        msg = "One or more cubes {} are missing".format(cubes) + \
              " time coordinate: {}".format(str(exc))
        raise ValueError(msg)
    except TypeError as error:
        msg = ("Cubes cannot be sorted "
               f"due to differing time units: {str(error)}")
        raise TypeError(msg) from error
    return cubes


def concatenate(cubes, check_level=CheckLevels.DEFAULT):
    """Concatenate all cubes after fixing metadata.

    Parameters
    ----------
    cubes: iterable of iris.cube.Cube
        Data cubes to be concatenated
    check_level: CheckLevels
        Level of strictness of the checks in the concatenation.

    Returns
    -------
    cube: iris.cube.Cube
        Resulting concatenated cube.

    Raises
    ------
    ValueError
        Concatenation was not possible.
    """
    if not cubes:
        return cubes
    if len(cubes) == 1:
        return cubes[0]

    merge_cube_attributes(cubes)
    cubes = _sort_cubes_by_time(cubes)
    _fix_calendars(cubes)
    cubes = _check_time_overlaps(cubes)
    result = _concatenate_cubes(cubes, check_level=check_level)

    if len(result) == 1:
        result = result[0]
    else:
        _get_concatenation_error(result)

    return result


def save(cubes,
         filename,
         optimize_access='',
         compress=False,
         alias='',
         **kwargs):
    """Save iris cubes to file.

    Parameters
    ----------
    cubes: iterable of iris.cube.Cube
        Data cubes to be saved

    filename: str
        Name of target file

    optimize_access: str
        Set internal NetCDF chunking to favour a reading scheme

        Values can be map or timeseries, which improve performance when
        reading the file one map or time series at a time.
        Users can also provide a coordinate or a list of coordinates. In that
        case the better performance will be avhieved by loading all the values
        in that coordinate at a time

    compress: bool, optional
        Use NetCDF internal compression.

    alias: str, optional
        Var name to use when saving instead of the one in the cube.

    Returns
    -------
    str
        filename

    Raises
    ------
    ValueError
        cubes is empty.
    """
    if not cubes:
        raise ValueError(f"Cannot save empty cubes '{cubes}'")

    # Rename some arguments
    kwargs['target'] = filename
    kwargs['zlib'] = compress

    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    if (os.path.exists(filename)
            and all(cube.has_lazy_data() for cube in cubes)):
        logger.debug(
            "Not saving cubes %s to %s to avoid data loss. "
            "The cube is probably unchanged.", cubes, filename)
        return filename

    for cube in cubes:
        logger.debug("Saving cube:\n%s\nwith %s data to %s", cube,
                     "lazy" if cube.has_lazy_data() else "realized", filename)
    if optimize_access:
        cube = cubes[0]
        if optimize_access == 'map':
            dims = set(
                cube.coord_dims('latitude') + cube.coord_dims('longitude'))
        elif optimize_access == 'timeseries':
            dims = set(cube.coord_dims('time'))
        else:
            dims = tuple()
            for coord_dims in (cube.coord_dims(dimension)
                               for dimension in optimize_access.split(' ')):
                dims += coord_dims
            dims = set(dims)

        kwargs['chunksizes'] = tuple(
            length if index in dims else 1
            for index, length in enumerate(cube.shape))

    kwargs['fill_value'] = GLOBAL_FILL_VALUE
    if alias:

        for cube in cubes:
            logger.debug('Changing var_name from %s to %s', cube.var_name,
                         alias)
            cube.var_name = alias

    # Ignore some warnings when saving
    with catch_warnings():
        filterwarnings(
            'ignore',
            message=(
                ".* is being added as CF data variable attribute, but .* "
                "should only be a CF global attribute"
            ),
            category=UserWarning,
            module='iris',
        )
        iris.save(cubes, **kwargs)

    return filename


def _get_debug_filename(filename, step):
    """Get a filename for debugging the preprocessor."""
    dirname = os.path.splitext(filename)[0]
    if os.path.exists(dirname) and os.listdir(dirname):
        num = int(sorted(os.listdir(dirname)).pop()[:2]) + 1
    else:
        num = 0
    filename = os.path.join(dirname, '{:02}_{}.nc'.format(num, step))
    return filename


def _sort_products(products):
    """Sort preprocessor output files by their order in the recipe."""
    return sorted(
        products,
        key=lambda p: (
            p.attributes.get('recipe_dataset_index', 1e6),
            p.attributes.get('dataset', ''),
        ),
    )


def write_metadata(products, write_ncl=False):
    """Write product metadata to file."""
    output_files = []
    for output_dir, prods in groupby(products,
                                     lambda p: os.path.dirname(p.filename)):
        sorted_products = _sort_products(prods)
        metadata = {}
        for product in sorted_products:
            if isinstance(product.attributes.get('exp'), (list, tuple)):
                product.attributes = dict(product.attributes)
                product.attributes['exp'] = '-'.join(product.attributes['exp'])
            if 'original_short_name' in product.attributes:
                del product.attributes['original_short_name']
            metadata[product.filename] = product.attributes

        output_filename = os.path.join(output_dir, 'metadata.yml')
        output_files.append(output_filename)
        with open(output_filename, 'w', encoding='utf-8') as file:
            yaml.safe_dump(metadata, file)
        if write_ncl:
            output_files.append(_write_ncl_metadata(output_dir, metadata))

    return output_files


def _write_ncl_metadata(output_dir, metadata):
    """Write NCL metadata files to output_dir."""
    variables = [copy.deepcopy(v) for v in metadata.values()]

    info = {'input_file_info': variables}

    # Split input_file_info into dataset and variable properties
    # dataset keys and keys with non-identical values will be stored
    # in dataset_info, the rest in variable_info
    variable_info = {}
    info['variable_info'] = [variable_info]
    info['dataset_info'] = []
    for variable in variables:
        dataset_info = {}
        info['dataset_info'].append(dataset_info)
        for key in variable:
            dataset_specific = any(variable[key] != var.get(key, object())
                                   for var in variables)
            if ((dataset_specific or key in DATASET_KEYS)
                    and key not in VARIABLE_KEYS):
                dataset_info[key] = variable[key]
            else:
                variable_info[key] = variable[key]

    filename = os.path.join(output_dir,
                            variable_info['short_name'] + '_info.ncl')
    write_ncl_settings(info, filename)

    return filename
