"""Functions for loading and saving cubes."""
import copy
import logging
import os
import shutil
from itertools import groupby
from warnings import catch_warnings, filterwarnings

import iris
import iris.aux_factory
import iris.exceptions
import numpy as np
import yaml
from cf_units import suppress_errors

from .._task import write_ncl_settings
from ._time import extract_time

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
    _delete_attributes(
        raw_cube,
        ('creation_date', 'tracking_id', 'history', 'comment')
    )
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


def _fix_cube_attributes(cubes):
    """Unify attributes of different cubes to allow concatenation."""
    attributes = {}
    for cube in cubes:
        for (attr, val) in cube.attributes.items():
            if attr not in attributes:
                attributes[attr] = val
            else:
                if not np.array_equal(val, attributes[attr]):
                    attributes[attr] = '{};{}'.format(str(attributes[attr]),
                                                      str(val))
    for cube in cubes:
        cube.attributes = attributes


def _by_two_concatenation(cubes):
    """Perform a by-2 concatenation to avoid gaps."""
    concatenated = iris.cube.CubeList(cubes).concatenate()
    if len(concatenated) == 1:
        return concatenated[0]

    concatenated = _concatenate_overlapping_cubes(concatenated)
    if len(concatenated) == 2:
        _get_concatenation_error(concatenated)
    else:
        return concatenated[0]


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


def concatenate(cubes):
    """Concatenate all cubes after fixing metadata."""
    if not cubes:
        return cubes
    if len(cubes) == 1:
        return cubes[0]

    _fix_cube_attributes(cubes)

    if len(cubes) > 1:
        # order cubes by first time point
        try:
            cubes = sorted(cubes, key=lambda c: c.coord("time").cell(0).point)
        except iris.exceptions.CoordinateNotFoundError as exc:
            msg = "One or more cubes {} are missing".format(cubes) + \
                  " time coordinate: {}".format(str(exc))
            raise ValueError(msg)

        # iteratively concatenate starting with first cube
        result = cubes[0]
        for cube in cubes[1:]:
            result = _by_two_concatenation([result, cube])

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

    logger.debug("Saving cubes %s to %s", cubes, filename)
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
    """Clean up after running the preprocessor."""
    if remove is None:
        remove = []

    for path in remove:
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.isfile(path):
            os.remove(path)

    return files


def write_metadata(products, write_ncl=False):
    """Write product metadata to file."""
    output_files = []
    for output_dir, prods in groupby(products,
                                     lambda p: os.path.dirname(p.filename)):
        sorted_products = sorted(
            prods,
            key=lambda p: (
                p.attributes.get('recipe_dataset_index', 1e6),
                p.attributes.get('dataset', ''),
            ),
        )
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


def _concatenate_overlapping_cubes(cubes):
    """Concatenate time-overlapping cubes (two cubes only)."""
    # we arrange [cube1, cube2] so that cube1.start <= cube2.start
    if cubes[0].coord('time').points[0] <= cubes[1].coord('time').points[0]:
        cubes = [cubes[0], cubes[1]]
        logger.debug(
            "Will attempt to concatenate cubes %s "
            "and %s in this order", cubes[0], cubes[1])
    else:
        cubes = [cubes[1], cubes[0]]
        logger.debug(
            "Will attempt to concatenate cubes %s "
            "and %s in this order", cubes[1], cubes[0])

    # get time end points
    time_1 = cubes[0].coord('time')
    time_2 = cubes[1].coord('time')
    if time_1.units != time_2.units:
        raise ValueError(
            f"Cubes\n{cubes[0]}\nand\n{cubes[1]}\ncan not be concatenated: "
            f"time units {time_1.units}, calendar {time_1.units.calendar} "
            f"and {time_2.units}, calendar {time_2.units.calendar} differ")
    data_start_1 = time_1.cell(0).point
    data_start_2 = time_2.cell(0).point
    data_end_1 = time_1.cell(-1).point
    data_end_2 = time_2.cell(-1).point

    # case 1: both cubes start at the same time -> return longer cube
    if data_start_1 == data_start_2:
        if data_end_1 <= data_end_2:
            logger.debug(
                "Both cubes start at the same time but cube %s "
                "ends before %s", cubes[0], cubes[1])
            logger.debug("Cube %s contains all needed data so using it fully",
                         cubes[1])
            cubes = [cubes[1]]
        else:
            logger.debug(
                "Both cubes start at the same time but cube %s "
                "ends before %s", cubes[1], cubes[0])
            logger.debug("Cube %s contains all needed data so using it fully",
                         cubes[0])
            cubes = [cubes[0]]

    # case 2: cube1 starts before cube2
    else:
        # find time overlap, if any
        start_overlap = next((time_1.units.num2date(t)
                              for t in time_1.points if t in time_2.points),
                             None)
        # case 2.0: no overlap (new iris implementation does allow
        # concatenation of cubes with no overlap)
        if not start_overlap:
            logger.debug(
                "Unable to concatenate non-overlapping cubes\n%s\nand\n%s"
                "separated in time.", cubes[0], cubes[1])
        # case 2.1: cube1 ends after cube2 -> return cube1
        elif data_end_1 > data_end_2:
            cubes = [cubes[0]]
            logger.debug("Using only data from %s", cubes[0])
        # case 2.2: cube1 ends before cube2 -> use full cube2 and shorten cube1
        else:
            logger.debug(
                "Extracting time slice between %s and %s from cube %s to use "
                "it for concatenation with cube %s", "-".join([
                    str(data_start_1.year),
                    str(data_start_1.month),
                    str(data_start_1.day)
                ]), "-".join([
                    str(start_overlap.year),
                    str(start_overlap.month),
                    str(start_overlap.day)
                ]), cubes[0], cubes[1])
            c1_delta = extract_time(cubes[0], data_start_1.year,
                                    data_start_1.month, data_start_1.day,
                                    start_overlap.year, start_overlap.month,
                                    start_overlap.day)
            # convert c1_delta scalar cube to vector cube, if needed
            if c1_delta.data.shape == ():
                c1_delta = iris.util.new_axis(c1_delta, scalar_coord="time")
            cubes = iris.cube.CubeList([c1_delta, cubes[1]])
            logger.debug("Attempting concatenatenation of %s with %s",
                         c1_delta, cubes[1])
            try:
                cubes = [iris.cube.CubeList(cubes).concatenate_cube()]
            except iris.exceptions.ConcatenateError as ex:
                logger.error('Can not concatenate cubes: %s', ex)
                logger.error('Cubes:')
                for cube in cubes:
                    logger.error(cube)
                raise ex

    return cubes
