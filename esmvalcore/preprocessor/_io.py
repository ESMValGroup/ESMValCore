"""Functions for loading and saving cubes."""
import copy
import datetime
import logging
import os
import shutil
from collections import OrderedDict, namedtuple
from itertools import groupby

import iris
import iris.coord_categorisation
import iris.exceptions
import yaml
from cf_units import Unit

from esmvalcore.cmor.check import check_frequency
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

TimeRange = namedtuple('TimeRange', ['start', 'end'])


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
    for attr in ['creation_date', 'tracking_id', 'history']:
        if attr in raw_cube.attributes:
            del raw_cube.attributes[attr]
    for coord in raw_cube.coords():
        # Iris chooses to change longitude and latitude units to degrees
        # regardless of value in file, so reinstating file value
        if coord.standard_name in ['longitude', 'latitude']:
            units = _get_attr_from_field_coord(field, coord.var_name, 'units')
            if units is not None:
                coord.units = units


def load(file, callback=None):
    """Load iris cubes from files."""
    logger.debug("Loading:\n%s", file)
    raw_cubes = iris.load_raw(file, callback=callback)
    if not raw_cubes:
        raise Exception('Can not load cubes from {0}'.format(file))
    for cube in raw_cubes:
        cube.attributes['source_file'] = file
    return raw_cubes


def fix_cube_attributes(cubes):
    """Unify attributes of different cubes to allow concatenation."""
    attributes = {}
    for cube in cubes:
        for (attr, val) in cube.attributes.items():
            if attr not in attributes:
                attributes[attr] = val
            else:
                if str(val) not in str(attributes[attr]):
                    attributes[attr] = '{}|{}'.format(
                        str(attributes[attr]), str(val))
    for cube in cubes:
        cube.attributes = attributes


def concatenate(cubes):
    """Concatenate all cubes after fixing metadata."""
    fix_cube_attributes(cubes)
    try:
        cube = iris.cube.CubeList(cubes).concatenate_cube()
        return cube
    except iris.exceptions.ConcatenateError as ex:
        logger.error('Can not concatenate cubes: %s', ex)
        logger.error('Cubes:')
        for cube in cubes:
            logger.error(cube)
        raise ex

def _add_aux_time_coords(cube):
    """Add auxiliary time coordinates to cube."""
    coords = [coord.name() for coord in cube.aux_coords]
    if 'day_of_month' not in coords:
        iris.coord_categorisation.add_day_of_month(cube, 'time')
    if 'day_of_year' not in coords:
        iris.coord_categorisation.add_day_of_year(cube, 'time')
    if 'month_number' not in coords:
        iris.coord_categorisation.add_month_number(cube, 'time')
    if 'year' not in coords:
        iris.coord_categorisation.add_year(cube, 'time')


def _extract_time_range(cube):
    """Extract time range of a cube."""
    time_coord = cube.coord('time')
    start = time_coord.units.num2date(time_coord.points[0])
    end = time_coord.units.num2date(time_coord.points[-1])
    return TimeRange(start=start, end=end)


def _to_iris_partial_datetime(time_object):
    """"Convert :mod:`datetime` object to :mod:`iris.time.PartialDateTime`."""
    return iris.time.PartialDateTime(
        year=time_object.year,
        month=time_object.month,
        day=time_object.day,
        hour=time_object.hour,
        minute=time_object.minute,
        second=time_object.second,
        microsecond=time_object.microsecond,
    )


def _fix_cube_metadata(cubes):
    """Fix metadata of cubes prior to concatenation."""
    fix_cube_attributes(cubes)

    # Auxiliary time units
    for cube in cubes:
        time_idx = cube.coord_dims('time')
        for coord in cube.coords(dim_coords=False,
                                 contains_dimension=time_idx[0]):
            cube.remove_coord(coord)

    # Units of latitude and longitude
    degrees_units = Unit('degrees')
    for cube in cubes:
        for coord in cube.coords():
            if coord.units == degrees_units:
                coord.units = degrees_units
                if coord.name() == 'longitude':
                    coord.circular = True


def _realize_cube(cube):
    """Realize cube to memory in order to avoid loss of data."""
    cube.data
    for coord in cube.coords():
        coord.points
        coord.bounds


def _concatenate_along_time(old_cube, new_cube):
    """Check consistency and concatenate two cubes along time dimension."""
    old_time_range = _extract_time_range(old_cube)
    new_time_range = _extract_time_range(new_cube)

    # Check if time ranges overlap
    latest_start = max(old_time_range.start, new_time_range.start)
    earliest_end = min(old_time_range.end, new_time_range.end)
    delta = earliest_end - latest_start
    if delta > datetime.timedelta():
        logger.warning(
            "Old and new cubes overlap in time dimension, got %s for old cube "
            "and %s for new cube", old_time_range, new_time_range)
        start = _to_iris_partial_datetime(new_time_range[0])
        end = _to_iris_partial_datetime(new_time_range[1])
        time_constraint = iris.Constraint(
            time=lambda cell: cell.point < start or cell.point > end)
        old_cube = old_cube.extract(time_constraint)
        if old_cube is None:
            logger.warning("Overwriting all data of old cube with new cube")
            return new_cube
        else:
            logger.warning(
                "Overwriting parts of old cube with data of new cube")

    # Build final cube and realize data to avoid data loss
    cubes = iris.cube.CubeList([old_cube, new_cube])
    _fix_cube_metadata(cubes)
    final_cube = cubes.concatenate_cube(check_aux_coords=False)
    _add_aux_time_coords(final_cube)
    _realize_cube(final_cube)

    # Check time frequencies (i.e. if data is missing)
    if 'frequency' in final_cube.attributes:
        time_coord = final_cube.coord('time')
        frequency = final_cube.attributes['frequency']
        (successful, msg) = check_frequency(time_coord, frequency)
        if not successful:
            logger.warning(
                msg.format(final_cube.summary(shorten=True), frequency))
            logger.warning(
                "Frequency check of final cube was not successful, the cube "
                "might not be contiguous")
    else:
        logger.warning(
            "Could not check if final cube is contiguous, cube attributes do "
            "not contain 'frequency'")

    return final_cube


def _save_new_cube_individually(cubes, kwargs, msg):
    """Save new cube individually in concatenation mode in case of errors."""
    kwargs = dict(kwargs)
    now = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    kwargs['target'] = kwargs['target'].replace('.nc', f'_{now}.nc')
    logger.warning(msg)
    logger.warning("Saving cubes %s to %s", cubes, kwargs['target'])
    iris.save(cubes, **kwargs)


def _concatenate_output(cubes, filename, **kwargs):
    """Concatenate cubes to already existing output."""
    logger.debug("Saving cubes in concatenation mode")
    if not os.path.isfile(filename):
        logger.debug(
            "File %s does not exits yet, saving cubes %s", filename, cubes)
        iris.save(cubes, **kwargs)
        return
    if len(cubes) > 1:
        msg = (f"Saving cubes in concatenation mode is not possible for "
               f"CubeLists with more than one element, got {len(cubes)}")
        return _save_new_cube_individually(cubes, kwargs, msg)

    # If file exists, check for consistency and concatenate
    old_cubes = iris.load(filename)
    if len(old_cubes) > 1:
        msg = (f"Saving cubes in concatenation mode is not possible if old "
               f"file at {filename} contains a CubeList with more than one "
               f"element, got {len(cubes)}")
        return _save_new_cube_individually(cubes, kwargs, msg)

    # Consistency check and concatenation
    try:
        new_cube = _concatenate_along_time(old_cubes[0], cubes[0])
    except Exception as exc:
        msg = f"Could not concatenate old and new cube along time: {str(exc)}"
        return _save_new_cube_individually(cubes, kwargs, msg)

    #  Save final cube
    logger.info("Successfully concatenated cube %s to %s",
                new_cube.summary(shorten=True), filename)
    iris.save(new_cube, **kwargs)


def save(cubes, filename, optimize_access='', compress=False,
         concatenate=False, **kwargs):
    """
    Save iris cubes to file.

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

    concatenate: bool, optional
        Concatenate cubes to already existent output if possible (used in
        quicklook mode).

    Returns
    -------
    str
        filename

    """
    # Rename some arguments
    kwargs['target'] = filename
    kwargs['zlib'] = compress

    # Check if directory exits and create it if necessary
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # Configure keyword arguments
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

    # Concatenate output if desired
    if concatenate:
        _concatenate_output(cubes, filename, **kwargs)
        return filename

    # Save file regularly
    if (os.path.exists(filename)
        and all(cube.has_lazy_data() for cube in cubes)):
        logger.debug(
            "Not saving cubes %s to %s to avoid data loss. "
            "The cube is probably unchanged.", cubes, filename)
        return filename
    logger.debug("Saving cubes %s to %s", cubes, filename)
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


def _ordered_safe_dump(data, stream):
    """Write data containing OrderedDicts to yaml file."""

    class _OrderedDumper(yaml.SafeDumper):
        pass

    def _dict_representer(dumper, data):
        return dumper.represent_mapping(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, data.items())

    _OrderedDumper.add_representer(OrderedDict, _dict_representer)
    return yaml.dump(data, stream, _OrderedDumper)


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
        metadata = OrderedDict()
        for product in sorted_products:
            if isinstance(product.attributes.get('exp'), (list, tuple)):
                product.attributes = dict(product.attributes)
                product.attributes['exp'] = '-'.join(product.attributes['exp'])
            metadata[product.filename] = product.attributes

        output_filename = os.path.join(output_dir, 'metadata.yml')
        output_files.append(output_filename)
        with open(output_filename, 'w') as file:
            _ordered_safe_dump(metadata, file)
        if write_ncl:
            output_files.append(_write_ncl_metadata(output_dir, metadata))

    return output_files


def _write_ncl_metadata(output_dir, metadata):
    """Write NCL metadata files to output_dir."""
    variables = [copy.deepcopy(v) for v in metadata.values()]

    for variable in variables:
        fx_files = variable.pop('fx_files', {})
        for fx_type in fx_files:
            variable[fx_type] = fx_files[fx_type]

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
            dataset_specific = any(
                variable[key] != var.get(key, object()) for var in variables)
            if ((dataset_specific or key in DATASET_KEYS)
                    and key not in VARIABLE_KEYS):
                dataset_info[key] = variable[key]
            else:
                variable_info[key] = variable[key]

    filename = os.path.join(output_dir,
                            variable_info['short_name'] + '_info.ncl')
    write_ncl_settings(info, filename)

    return filename
