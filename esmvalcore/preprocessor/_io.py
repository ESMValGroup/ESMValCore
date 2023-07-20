"""Functions for loading and saving cubes."""
import copy
import logging
import os
import shutil
import warnings
from itertools import groupby
from warnings import catch_warnings, filterwarnings

import iris
import iris.aux_factory
import iris.exceptions
import isodate
import numpy as np
import yaml
from cf_units import suppress_errors

from esmvalcore.cmor.check import CheckLevels
from esmvalcore.exceptions import ESMValCoreDeprecationWarning
from esmvalcore.iris_helpers import merge_cube_attributes

from .._task import write_ncl_settings
from ._time import clip_timerange

logger = logging.getLogger(__name__)

GLOBAL_FILL_VALUE = 1e+20

DATASET_KEYS = {
    'mip',
}
VARIABLE_KEYS = {
    'reference_dataset',
    'alternative_dataset',
}


def _fix_aux_factories(cube):
    """Fix :class:`iris.aux_factory.AuxCoordFactory` after concatenation.

    Necessary because of bug in :mod:`iris` (see issue #2478).
    """
    coord_names = [coord.name() for coord in cube.coords()]

    # Hybrid sigma pressure coordinate
    # TODO possibly add support for other hybrid coordinates
    if 'atmosphere_hybrid_sigma_pressure_coordinate' in coord_names:
        new_aux_factory = iris.aux_factory.HybridPressureFactory(
            delta=cube.coord(var_name='ap'),
            sigma=cube.coord(var_name='b'),
            surface_air_pressure=cube.coord(var_name='ps'),
        )
        for aux_factory in cube.aux_factories:
            if isinstance(aux_factory, iris.aux_factory.HybridPressureFactory):
                break
        else:
            cube.add_aux_factory(new_aux_factory)

    # Hybrid sigma height coordinate
    if 'atmosphere_hybrid_height_coordinate' in coord_names:
        new_aux_factory = iris.aux_factory.HybridHeightFactory(
            delta=cube.coord(var_name='lev'),
            sigma=cube.coord(var_name='b'),
            orography=cube.coord(var_name='orog'),
        )
        for aux_factory in cube.aux_factories:
            if isinstance(aux_factory, iris.aux_factory.HybridHeightFactory):
                break
        else:
            cube.add_aux_factory(new_aux_factory)

    # Atmosphere sigma coordinate
    if 'atmosphere_sigma_coordinate' in coord_names:
        new_aux_factory = iris.aux_factory.AtmosphereSigmaFactory(
            pressure_at_top=cube.coord(var_name='ptop'),
            sigma=cube.coord(var_name='lev'),
            surface_air_pressure=cube.coord(var_name='ps'),
        )
        for aux_factory in cube.aux_factories:
            if isinstance(aux_factory,
                          iris.aux_factory.AtmosphereSigmaFactory):
                break
        else:
            cube.add_aux_factory(new_aux_factory)


def _get_attr_from_field_coord(ncfield, coord_name, attr):
    if coord_name is not None:
        attrs = ncfield.cf_group[coord_name].cf_attrs()
        attr_val = [value for (key, value) in attrs if key == attr]
        if attr_val:
            return attr_val[0]
    return None


def concatenate_callback(raw_cube, field, _):
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


def load(file, callback=None, ignore_warnings=None):
    """Load iris cubes from files.

    Parameters
    ----------
    file: str
        File to be loaded.
    callback: callable or None, optional (default: None)
        Callback function passed to :func:`iris.load_raw`.

        .. deprecated:: 2.8.0
            This argument will be removed in 2.10.0.
    ignore_warnings: list of dict or None, optional (default: None)
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
    if not (callback is None or callback == 'default'):
        msg = ("The argument `callback` has been deprecated in "
               "ESMValCore version 2.8.0 and is scheduled for removal in "
               "version 2.10.0.")
        warnings.warn(msg, ESMValCoreDeprecationWarning)
    if callback == 'default':
        callback = concatenate_callback
    file = str(file)
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
    })

    # Filter warnings
    with catch_warnings():
        for warning_kwargs in ignore_warnings:
            warning_kwargs.setdefault('action', 'ignore')
            filterwarnings(**warning_kwargs)
        # Suppress UDUNITS-2 error messages that cannot be ignored with
        # warnings.filterwarnings
        # (see https://github.com/SciTools/cf-units/issues/240)
        with suppress_errors():
            raw_cubes = iris.load_raw(file, callback=callback)
    logger.debug("Done with loading %s", file)
    if not raw_cubes:
        raise ValueError(f'Can not load cubes from {file}')
    for cube in raw_cubes:
        cube.attributes['source_file'] = file
    return raw_cubes


def _concatenate_cubes(cubes, check_level):
    """Concatenate cubes according to the check_level."""

    kwargs = {
        'check_aux_coords': True,
        'check_cell_measures': True,
        'check_ancils': True,
        #    'check_derived_coords': True
    }

    if check_level > CheckLevels.DEFAULT:
        kwargs = dict.fromkeys(kwargs, False)
        logger.debug(
            'Concatenation will be performed without checking '
            'auxiliary coordinates, cell measures, ancillaries '
            'and derived coordinates present in the cubes.', )

    concatenated = iris.cube.CubeList(cubes).concatenate(**kwargs)
    if len(concatenated) == 1:
        return concatenated[0]
    else:
        _get_concatenation_error(concatenated)


def _check_time_overlaps(cubes):
    """Handle time overlaps."""
    units = set([cube.coord('time').units for cube in cubes])
    if len(units) > 1:
        raise ValueError(
            f"Cubes\n{cubes[0]}\nand\n{cubes[1]}\ncan not be concatenated: "
            f"time units in cubes differ")

    times = [cube.coord('time').core_points() for cube in cubes]
    for index, _ in enumerate(times[:-1]):
        overlap = np.intersect1d(times[index], times[index + 1])
        if overlap.shape != ():
            overlapping_cubes = cubes[index:index + 2]
            time_1 = overlapping_cubes[0].coord('time').core_points()
            time_2 = overlapping_cubes[1].coord('time').core_points()
            if time_1[0] == time_2[0]:
                if time_1[-1] > time_2[-1]:
                    cubes.pop(index + 1)
                    logger.debug(
                        "Both cubes start at the same time but cube %s "
                        "ends before")
                else:
                    cubes.pop(index)
                    logger.debug(
                        "Both cubes start at the same time but cube %s "
                        "ends before")
            elif time_1[-1] > time_2[-1]:
                cubes.pop(index + 1)
            else:
                new_time = np.delete(time_1,
                                     np.argwhere(np.in1d(time_1, overlap)))
                new_dates = overlapping_cubes[0].coord('time').units.num2date(
                    new_time)

                start_point = isodate.date_isoformat(
                    new_dates[0], format=isodate.isostrf.DATE_BAS_COMPLETE)
                end_point = isodate.date_isoformat(
                    new_dates[-1], format=isodate.isostrf.DATE_BAS_COMPLETE)
                new_cube = clip_timerange(overlapping_cubes[0],
                                          f'{start_point}/{end_point}')
                if new_cube.shape == ():
                    new_cube = iris.util.new_axis(new_cube,
                                                  scalar_coord="time")

                cubes[index] = new_cube
    return cubes


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
    """Sort CubeList by time coordinate"""
    try:
        cubes = sorted(cubes, key=lambda c: c.coord("time").cell(0).point)
    except iris.exceptions.CoordinateNotFoundError as exc:
        msg = "One or more cubes {} are missing".format(cubes) + \
              " time coordinate: {}".format(str(exc))
        raise ValueError(msg)
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
    cubes = _check_time_overlaps(cubes)
    result = _concatenate_cubes(cubes, check_level=check_level)

    _fix_aux_factories(result)

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


def cleanup(files, remove=None):
    """Clean up after running the preprocessor.

    Warning
    -------
    .. deprecated:: 2.8.0
        This function is no longer used and has been deprecated since
        ESMValCore version 2.8.0. It is scheduled for removal in version
        2.10.0.

    Parameters
    ----------
    files: list of Path
        Preprocessor output files (will not be removed if not in `removed`).
    remove: list of Path or None, optional (default: None)
        Files or directories to remove.

    Returns
    -------
    list of Path
        Preprocessor output files.
    """
    deprecation_msg = (
        "The preprocessor function `cleanup` has been deprecated in "
        "ESMValCore version 2.8.0 and is scheduled for removal in version "
        "2.10.0.")
    warnings.warn(deprecation_msg, ESMValCoreDeprecationWarning)

    if remove is None:
        remove = []

    for path in remove:
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.isfile(path):
            os.remove(path)

    return files


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
        with open(output_filename, 'w') as file:
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
