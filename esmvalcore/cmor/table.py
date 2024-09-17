"""CMOR information reader for ESMValTool.

Read variable information from CMOR 2 and CMOR 3 tables and make it
easily available for the other components of ESMValTool
"""
from __future__ import annotations

import copy
import errno
import glob
import json
import logging
import os
from collections import Counter
from functools import lru_cache, total_ordering
from pathlib import Path
from typing import Optional, Union

import yaml

from esmvalcore.exceptions import RecipeError

logger = logging.getLogger(__name__)

CMORTable = Union['CMIP3Info', 'CMIP5Info', 'CMIP6Info', 'CustomInfo']

CMOR_TABLES: dict[str, CMORTable] = {}
"""dict of str, obj: CMOR info objects."""

_CMOR_KEYS = (
    'standard_name',
    'long_name',
    'units',
    'modeling_realm',
    'frequency',
)


def _update_cmor_facets(facets):
    """Update `facets` with information from CMOR table."""
    project = facets['project']
    mip = facets['mip']
    short_name = facets['short_name']
    derive = facets.get('derive', False)
    table = CMOR_TABLES.get(project)
    if table:
        table_entry = table.get_variable(mip, short_name, derive)
    else:
        table_entry = None
    if table_entry is None:
        raise RecipeError(
            f"Unable to load CMOR table (project) '{project}' for variable "
            f"'{short_name}' with mip '{mip}'")
    facets['original_short_name'] = table_entry.short_name
    for key in _CMOR_KEYS:
        if key not in facets:
            value = getattr(table_entry, key, None)
            if value is not None:
                facets[key] = value
            else:
                logger.debug(
                    "Failed to add key %s to variable %s from CMOR table", key,
                    facets)


def _get_mips(project: str, short_name: str) -> list[str]:
    """Get all available MIP tables in a project."""
    tables = CMOR_TABLES[project].tables
    mips = [mip for mip in tables if short_name in tables[mip]]
    return mips


def get_var_info(
    project: str,
    mip: str,
    short_name: str,
) -> VariableInfo | None:
    """Get variable information.

    Note
    ----
    If `project=CORDEX` and the `mip` ends with 'hr', it is cropped to 'h'
    since CORDEX X-hourly tables define the `mip` as ending in 'h' instead of
    'hr'.

    Parameters
    ----------
    project:
        Dataset's project.
    mip:
        Variable's CMOR table, i.e., MIP.
    short_name:
        Variable's short name.

    Returns
    -------
    VariableInfo | None
        `VariableInfo` object for the requested variable if found, ``None``
        otherwise.

    Raises
    ------
    KeyError
        No CMOR tables available for `project`.

    """
    if project not in CMOR_TABLES:
        raise KeyError(
            f"No CMOR tables available for project '{project}'. The following "
            f"tables are available: {', '.join(CMOR_TABLES)}."
        )

    # CORDEX X-hourly tables define the mip as ending in 'h' instead of 'hr'
    if project == 'CORDEX' and mip.endswith('hr'):
        mip = mip.replace('hr', 'h')

    return CMOR_TABLES[project].get_variable(mip, short_name)


def read_cmor_tables(cfg_developer: Optional[Path] = None) -> None:
    """Read cmor tables required in the configuration.

    Parameters
    ----------
    cfg_developer:
        Path to config-developer.yml file.

    Raises
    ------
    TypeError
        If `cfg_developer` is not a Path-like object
    """
    if cfg_developer is None:
        cfg_developer = Path(__file__).parents[1] / 'config-developer.yml'
    elif not isinstance(cfg_developer, Path):
        raise TypeError("cfg_developer is not a Path-like object, got ",
                        cfg_developer)
    mtime = cfg_developer.stat().st_mtime
    cmor_tables = _read_cmor_tables(cfg_developer, mtime)
    CMOR_TABLES.clear()
    CMOR_TABLES.update(cmor_tables)


@lru_cache
def _read_cmor_tables(cfg_file: Path, mtime: float) -> dict[str, CMORTable]:
    """Read cmor tables required in the configuration.

    Parameters
    ----------
    cfg_file: pathlib.Path
        Path to config-developer.yml file.
    mtime: float
        Modification time of config-developer.yml file. Only used by the
        `lru_cache` decorator to make sure the file is read again when it
        is changed.
    """
    with cfg_file.open('r', encoding='utf-8') as file:
        cfg_developer = yaml.safe_load(file)
    cwd = os.path.dirname(os.path.realpath(__file__))
    var_alt_names_file = os.path.join(cwd, 'variable_alt_names.yml')
    with open(var_alt_names_file, 'r', encoding='utf-8') as yfile:
        alt_names = yaml.safe_load(yfile)

    cmor_tables: dict[str, CMORTable] = {}

    # Try to infer location for custom tables from config-developer.yml file,
    # if not possible, use default location
    custom_path = None
    if 'custom' in cfg_developer:
        custom_path = cfg_developer['custom'].get('cmor_path')
    if custom_path is not None:
        custom_path = os.path.expandvars(os.path.expanduser(custom_path))
    custom = CustomInfo(custom_path)
    cmor_tables['custom'] = custom

    install_dir = os.path.dirname(os.path.realpath(__file__))
    for table in cfg_developer:
        if table == 'custom':
            continue
        cmor_tables[table] = _read_table(cfg_developer, table, install_dir,
                                         custom, alt_names)
    return cmor_tables


def _read_table(cfg_developer, table, install_dir, custom, alt_names):
    project = cfg_developer[table]
    cmor_type = project.get('cmor_type', 'CMIP5')
    default_path = os.path.join(install_dir, 'tables', cmor_type.lower())
    table_path = project.get('cmor_path', default_path)
    table_path = os.path.expandvars(os.path.expanduser(table_path))
    cmor_strict = project.get('cmor_strict', True)
    default_table_prefix = project.get('cmor_default_table_prefix', '')

    if cmor_type == 'CMIP3':
        return CMIP3Info(
            table_path,
            default=custom,
            strict=cmor_strict,
            alt_names=alt_names,
        )

    if cmor_type == 'CMIP5':
        return CMIP5Info(table_path,
                         default=custom,
                         strict=cmor_strict,
                         alt_names=alt_names)

    if cmor_type == 'CMIP6':
        return CMIP6Info(
            table_path,
            default=custom,
            strict=cmor_strict,
            default_table_prefix=default_table_prefix,
            alt_names=alt_names,
        )
    raise ValueError(f'Unsupported CMOR type {cmor_type}')


class InfoBase():
    """Base class for all table info classes.

    This uses CMOR 3 json format

    Parameters
    ----------
    default: object
        Default table to look variables on if not found

    alt_names: list[list[str]]
        List of known alternative names for variables

    strict: bool
        If False, will look for a variable in other tables if it can not be
        found in the requested one
    """

    def __init__(self, default, alt_names, strict):
        if alt_names is None:
            alt_names = ""
        self.default = default
        self.alt_names = alt_names
        self.strict = strict
        self.tables = {}

    def get_table(self, table):
        """Search and return the table info.

        Parameters
        ----------
        table: str
            Table name

        Returns
        -------
        TableInfo
            Return the TableInfo object for the requested table if
            found, returns None if not
        """
        return self.tables.get(table)

    def get_variable(
        self,
        table_name: str,
        short_name: str,
        derived: bool = False,
    ) -> VariableInfo | None:
        """Search and return the variable information.

        Parameters
        ----------
        table_name:
            Table name, i.e., the variable's MIP.
        short_name:
            Variable's short name.
        derived:
            Variable is derived. Information retrieval for derived variables
            always looks in the default tables (usually, the custom tables) if
            variable is not found in the requested table.

        Returns
        -------
        VariableInfo | None
            `VariableInfo` object for the requested variable if found, ``None``
            otherwise.

        """
        alt_names_list = self._get_alt_names_list(short_name)

        # First, look in requested table
        table = self.get_table(table_name)
        if table:
            for alt_names in alt_names_list:
                try:
                    return table[alt_names]
                except KeyError:
                    pass

        # If that didn't work, look in all tables (i.e., other MIPs) if
        # cmor_strict=False or derived=True
        var_info = self._look_in_all_tables(derived, alt_names_list)

        # If that didn't work either, look in default table if
        # cmor_strict=False or derived=True
        if not var_info:
            var_info = self._look_in_default(derived, alt_names_list,
                                             table_name)

        # If necessary, adapt frequency of variable (set it to the one from the
        # requested MIP). E.g., if the user asked for table `Amon`, but the
        # variable has been found in `day`, use frequency `mon`.
        if var_info:
            var_info = var_info.copy()
            var_info = self._update_frequency_from_mip(table_name, var_info)

        return var_info

    def _look_in_default(self, derived, alt_names_list, table_name):
        """Look for variable in default table."""
        var_info = None
        if (not self.strict or derived):
            for alt_names in alt_names_list:
                var_info = self.default.get_variable(table_name, alt_names)
                if var_info:
                    break
        return var_info

    def _look_in_all_tables(self, derived, alt_names_list):
        """Look for variable in all tables."""
        var_info = None
        if (not self.strict or derived):
            for alt_names in alt_names_list:
                var_info = self._look_all_tables(alt_names)
                if var_info:
                    break
        return var_info

    def _get_alt_names_list(self, short_name):
        """Get list of alternative variable names."""
        alt_names_list = [short_name]
        for alt_names in self.alt_names:
            if short_name in alt_names:
                alt_names_list.extend([
                    alt_name for alt_name in alt_names
                    if alt_name not in alt_names_list
                ])
        return alt_names_list

    def _update_frequency_from_mip(self, table_name, var_info):
        """Update frequency information of var_info from table."""
        mip_info = self.get_table(table_name)
        if mip_info:
            var_info.frequency = mip_info.frequency
        return var_info

    def _look_all_tables(self, alt_names):
        """Look for variable in all tables."""
        for table_vars in sorted(self.tables.values()):
            if alt_names in table_vars:
                return table_vars[alt_names]
        return None


class CMIP6Info(InfoBase):
    """Class to read CMIP6-like data request.

    This uses CMOR 3 json format

    Parameters
    ----------
    cmor_tables_path: str
        Path to the folder containing the Tables folder with the json files

    default: object
        Default table to look variables on if not found

    strict: bool
        If False, will look for a variable in other tables if it can not be
        found in the requested one
    """

    def __init__(self,
                 cmor_tables_path,
                 default=None,
                 alt_names=None,
                 strict=True,
                 default_table_prefix=''):

        super().__init__(default, alt_names, strict)
        cmor_tables_path = self._get_cmor_path(cmor_tables_path)

        self._cmor_folder = os.path.join(cmor_tables_path, 'Tables')
        if glob.glob(os.path.join(self._cmor_folder, '*_CV.json')):
            self._load_controlled_vocabulary()

        self.default_table_prefix = default_table_prefix

        self.var_to_freq = {}

        self._load_coordinates()
        for json_file in glob.glob(os.path.join(self._cmor_folder, '*.json')):
            if 'CV_test' in json_file or 'grids' in json_file:
                continue
            try:
                self._load_table(json_file)
            except Exception:
                msg = f"Exception raised when loading {json_file}"
                # Logger may not be ready at this stage
                if logger.handlers:
                    logger.error(msg)
                else:
                    print(msg)
                raise

    @staticmethod
    def _get_cmor_path(cmor_tables_path):
        if os.path.isdir(cmor_tables_path):
            return cmor_tables_path
        cwd = os.path.dirname(os.path.realpath(__file__))
        cmor_tables_path = os.path.join(cwd, 'tables', cmor_tables_path)
        if os.path.isdir(cmor_tables_path):
            return cmor_tables_path
        raise ValueError(
            'CMOR tables not found in {}'.format(cmor_tables_path))

    def _load_table(self, json_file):
        with open(json_file, encoding='utf-8') as inf:
            raw_data = json.loads(inf.read())
            if not self._is_table(raw_data):
                return
            table = TableInfo()
            header = raw_data['Header']
            table.name = header['table_id'].split(' ')[-1]
            self.tables[table.name] = table

            generic_levels = header['generic_levels'].split()
            table.frequency = header.get('frequency', '')
            self.var_to_freq[table.name] = {}

            for var_name, var_data in raw_data['variable_entry'].items():
                var = VariableInfo('CMIP6', var_name)
                var.read_json(var_data, table.frequency)
                self._assign_dimensions(var, generic_levels)
                table[var_name] = var
                self.var_to_freq[table.name][var_name] = var.frequency

            if not table.frequency:
                var_freqs = (var.frequency for var in table.values())
                table_freq, _ = Counter(var_freqs).most_common(1)[0]
                table.frequency = table_freq
            self.tables[table.name] = table

    def _assign_dimensions(self, var, generic_levels):
        for dimension in var.dimensions:
            if dimension in generic_levels:
                coord = CoordinateInfo(dimension)
                coord.generic_level = True
                for name in self.coords:
                    generic_level = self.coords[name].generic_lev_name
                    if dimension in [generic_level]:
                        coord.generic_lev_coords[name] = self.coords[name]
            else:
                try:
                    coord = self.coords[dimension]
                except KeyError:
                    logger.exception(
                        'Can not find dimension %s for variable %s', dimension,
                        var)
                    raise

            var.coordinates[dimension] = coord

    def _load_coordinates(self):
        self.coords = {}
        for json_file in glob.glob(
                os.path.join(self._cmor_folder, '*coordinate*.json')):
            with open(json_file, encoding='utf-8') as inf:
                table_data = json.loads(inf.read())
                for coord_name in table_data['axis_entry'].keys():
                    coord = CoordinateInfo(coord_name)
                    coord.read_json(table_data['axis_entry'][coord_name])
                    self.coords[coord_name] = coord

    def _load_controlled_vocabulary(self):
        self.activities = {}
        self.institutes = {}
        for json_file in glob.glob(os.path.join(self._cmor_folder,
                                                '*_CV.json')):
            with open(json_file, encoding='utf-8') as inf:
                table_data = json.loads(inf.read())
                try:
                    exps = table_data['CV']['experiment_id']
                    for exp_id in exps:
                        activity = exps[exp_id]['activity_id'][0].split(' ')
                        self.activities[exp_id] = activity
                except (KeyError, AttributeError):
                    pass

                try:
                    sources = table_data['CV']['source_id']
                    for source_id in sources:
                        institution = sources[source_id]['institution_id']
                        self.institutes[source_id] = institution
                except (KeyError, AttributeError):
                    pass

    def get_table(self, table):
        """Search and return the table info.

        Parameters
        ----------
        table: str
            Table name

        Returns
        -------
        TableInfo
            Return the TableInfo object for the requested table if
            found, returns None if not
        """
        try:
            return self.tables[table]
        except KeyError:
            return self.tables.get(''.join((self.default_table_prefix, table)))

    @staticmethod
    def _is_table(table_data):
        if 'variable_entry' not in table_data:
            return False
        if 'Header' not in table_data:
            return False
        return True


@total_ordering
class TableInfo(dict):
    """Container class for storing a CMOR table."""

    def __init__(self, *args, **kwargs):
        """Create a new TableInfo object for storing VariableInfo objects."""
        super(TableInfo, self).__init__(*args, **kwargs)
        self.name = ''
        self.frequency = ''
        self.realm = ''

    def __eq__(self, other):
        return (self.name, self.frequency, self.realm) == \
            (other.name, other.frequency, other.realm)

    def __ne__(self, other):
        return (self.name, self.frequency, self.realm) != \
            (other.name, other.frequency, other.realm)

    def __lt__(self, other):
        return (self.name, self.frequency, self.realm) < \
            (other.name, other.frequency, other.realm)


class JsonInfo(object):
    """Base class for the info classes.

    Provides common utility methods to read json variables
    """

    def __init__(self):
        self._json_data = {}

    def _read_json_variable(self, parameter, default=''):
        """Read a json parameter in json_data.

        Parameters
        ----------
        parameter: str
            parameter to read

        Returns
        -------
        str
            Option's value or empty string if parameter is not present
        """
        if parameter not in self._json_data:
            return default
        return str(self._json_data[parameter])

    def _read_json_list_variable(self, parameter):
        """Read a json list parameter in json_data.

        Parameters
        ----------
        parameter: str
            parameter to read

        Returns
        -------
        str
            Option's value or empty list if parameter is not present
        """
        if parameter not in self._json_data:
            return []
        return self._json_data[parameter]


class VariableInfo(JsonInfo):
    """Class to read and store variable information."""

    def __init__(self, table_type, short_name):
        """Class to read and store variable information.

        Parameters
        ----------
        short_name: str
            Variable's short name.
        """
        super(VariableInfo, self).__init__()
        self.table_type = table_type
        self.modeling_realm = []
        """Modeling realm"""
        self.short_name = short_name
        """Short name"""
        self.standard_name = ''
        """Standard name"""
        self.long_name = ''
        """Long name"""
        self.units = ''
        """Data units"""
        self.valid_min = ''
        """Minimum admitted value"""
        self.valid_max = ''
        """Maximum admitted value"""
        self.frequency = ''
        """Data frequency"""
        self.positive = ''
        """Increasing direction"""

        self.dimensions = []
        """List of dimensions"""
        self.coordinates = {}
        """Coordinates

        This is a dict with the names of the dimensions as keys and
        CoordinateInfo objects as values.
        """

        self._json_data = None

    def copy(self):
        """Return a shallow copy of VariableInfo.

        Returns
        -------
        VariableInfo
           Shallow copy of this object.
        """
        return copy.copy(self)

    def read_json(self, json_data, default_freq):
        """Read variable information from json.

        Non-present options will be set to empty

        Parameters
        ----------
        json_data: dict
            Dictionary created by the json reader containing variable
            information.
        default_freq: str
            Default frequency to use if it is not defined at variable level.
        """
        self._json_data = json_data

        self.standard_name = self._read_json_variable('standard_name')
        self.long_name = self._read_json_variable('long_name')
        self.units = self._read_json_variable('units')
        self.valid_min = self._read_json_variable('valid_min')
        self.valid_max = self._read_json_variable('valid_max')
        self.positive = self._read_json_variable('positive')
        self.modeling_realm = self._read_json_variable(
            'modeling_realm').split()
        self.frequency = self._read_json_variable('frequency', default_freq)

        self.dimensions = self._read_json_variable('dimensions').split()

    def has_coord_with_standard_name(self, standard_name: str) -> bool:
        """Check if a coordinate with a given `standard_name` exists.

        For some coordinates, multiple (slightly different) versions with
        different dimension names but identical `standard_name` exist. For
        example, the CMIP6 tables provide 4 different `standard_name=time`
        dimensions: `time`, `time1`, `time2`, and `time3`. Other examples would
        be the CMIP6 pressure levels (`plev19`, `plev23`, `plev27`, etc.  with
        standard name `air_pressure`) and the altitudes (`alt16`, `alt40` with
        standard name `altitude`).

        This function can be used to check for the existence of a specific
        coordinate defined by its `standard_name`, not its dimension name.

        Parameters
        ----------
        standard_name: str
            Standard name to be checked.

        Returns
        -------
        bool
            `True` if there is at least one coordinate with the given
            `standard_name`, `False` if not.

        """
        for coord in self.coordinates.values():
            if coord.standard_name == standard_name:
                return True
        return False


class CoordinateInfo(JsonInfo):
    """Class to read and store coordinate information."""

    def __init__(self, name):
        """Class to read and store coordinate information.

        Parameters
        ----------
        name: str
            coordinate's name
        """
        super(CoordinateInfo, self).__init__()
        self.name = name
        self.generic_level = False
        self.generic_lev_coords = {}

        self.axis = ""
        """Axis"""
        self.value = ""
        """Coordinate value"""
        self.standard_name = ""
        """Standard name"""
        self.long_name = ""
        """Long name"""
        self.out_name = ""
        """
        Out name

        This is the name of the variable in the file
        """
        self.var_name = ""
        """Short name"""
        self.units = ""
        """Units"""
        self.stored_direction = ""
        """Direction in which the coordinate increases"""
        self.requested = []
        """Values requested"""
        self.valid_min = ""
        """Minimum allowed value"""
        self.valid_max = ""
        """Maximum allowed value"""
        self.must_have_bounds = ""
        """Whether bounds are required on this dimension"""
        self.generic_lev_name = ""
        """Generic level name"""

    def read_json(self, json_data):
        """Read coordinate information from json.

        Non-present options will be set to empty

        Parameters
        ----------
        json_data: dict
            dictionary created by the json reader containing
            coordinate information
        """
        self._json_data = json_data

        self.axis = self._read_json_variable('axis')
        self.value = self._read_json_variable('value')
        self.out_name = self._read_json_variable('out_name')
        self.var_name = self._read_json_variable('var_name')
        self.standard_name = self._read_json_variable('standard_name')
        self.long_name = self._read_json_variable('long_name')
        self.units = self._read_json_variable('units')
        self.stored_direction = self._read_json_variable('stored_direction')
        self.valid_min = self._read_json_variable('valid_min')
        self.valid_max = self._read_json_variable('valid_max')
        self.requested = self._read_json_list_variable('requested')
        self.must_have_bounds = self._read_json_variable('must_have_bounds')
        self.generic_lev_name = self._read_json_variable('generic_level_name')


class CMIP5Info(InfoBase):
    """Class to read CMIP5-like data request.

    Parameters
    ----------
    cmor_tables_path: str
       Path to the folder containing the Tables folder with the json files

    default: object
        Default table to look variables on if not found

    strict: bool
        If False, will look for a variable in other tables if it can not be
        found in the requested one
    """

    def __init__(self,
                 cmor_tables_path,
                 default=None,
                 alt_names=None,
                 strict=True):
        super().__init__(default, alt_names, strict)
        cmor_tables_path = self._get_cmor_path(cmor_tables_path)

        self._cmor_folder = os.path.join(cmor_tables_path, 'Tables')
        if not os.path.isdir(self._cmor_folder):
            raise OSError(errno.ENOTDIR, "CMOR tables path is not a directory",
                          self._cmor_folder)

        self.strict = strict
        self.tables = {}
        self.coords = {}
        self._current_table = None
        self._last_line_read = None

        for table_file in glob.glob(os.path.join(self._cmor_folder, '*')):
            if '_grids' in table_file:
                continue
            try:
                self._load_table(table_file)
            except Exception:
                msg = f"Exception raised when loading {table_file}"
                # Logger may not be ready at this stage
                if logger.handlers:
                    logger.error(msg)
                else:
                    print(msg)
                raise

    @staticmethod
    def _get_cmor_path(cmor_tables_path):
        if os.path.isdir(cmor_tables_path):
            return cmor_tables_path
        cwd = os.path.dirname(os.path.realpath(__file__))
        cmor_tables_path = os.path.join(cwd, 'tables', cmor_tables_path)
        return cmor_tables_path

    def _load_table(self, table_file, table_name=''):
        if table_name and table_name in self.tables:
            # special case used for updating a table with custom variable file
            table = self.tables[table_name]
        else:
            # default case: table name is first line of table file
            table = None

        self._read_table_file(table_file, table)

    def _read_table_file(self, table_file, table=None):
        with open(table_file, 'r', encoding='utf-8') as self._current_table:
            self._read_line()
            while True:
                key, value = self._last_line_read
                if key == 'table_id':
                    table = TableInfo()
                    table.name = value[len('Table '):]
                    self.tables[table.name] = table
                elif key == 'frequency':
                    table.frequency = value
                elif key == 'modeling_realm':
                    table.realm = value
                elif key == 'generic_levels':
                    for dim in value.split(' '):
                        coord = CoordinateInfo(dim)
                        coord.generic_level = True
                        coord.axis = 'Z'
                        self.coords[dim] = coord
                elif key == 'axis_entry':
                    self.coords[value] = self._read_coordinate(value)
                    continue
                elif key == 'variable_entry':
                    table[value] = self._read_variable(value, table.frequency)
                    continue
                if not self._read_line():
                    return

    def _read_line(self):
        line = self._current_table.readline()
        if line == '':
            return False
        if line.startswith('!'):
            return self._read_line()
        line = line.replace('\n', '')
        if '!' in line:
            line = line[:line.index('!')]
        line = line.strip()
        if not line:
            self._last_line_read = ('', '')
        else:
            index = line.index(':')
            self._last_line_read = (line[:index].strip(),
                                    line[index + 1:].strip())
        return True

    def _read_coordinate(self, value):
        coord = CoordinateInfo(value)
        while self._read_line():
            key, value = self._last_line_read
            if key in ('variable_entry', 'axis_entry'):
                return coord
            if key == 'requested':
                coord.requested.extend(
                    (val for val in value.split(' ') if val))
                continue
            if hasattr(coord, key):
                setattr(coord, key, value)
        return coord

    def _read_variable(self, short_name, frequency):
        var = VariableInfo('CMIP5', short_name)
        var.frequency = frequency
        while self._read_line():
            key, value = self._last_line_read
            if key in ('variable_entry', 'axis_entry'):
                break
            if key in ('dimensions', 'modeling_realm'):
                setattr(var, key, value.split())
            elif hasattr(var, key):
                setattr(var, key, value)
        for dim in var.dimensions:
            var.coordinates[dim] = self.coords[dim]
        return var

    def get_table(self, table):
        """Search and return the table info.

        Parameters
        ----------
        table: str
            Table name

        Returns
        -------
        TableInfo
            Return the TableInfo object for the requested table if
            found, returns None if not
        """
        return self.tables.get(table)


class CMIP3Info(CMIP5Info):
    """Class to read CMIP3-like data request.

    Parameters
    ----------
    cmor_tables_path: str
       Path to the folder containing the Tables folder with the json files

    default: object
        Default table to look variables on if not found

    strict: bool
        If False, will look for a variable in other tables if it can not be
        found in the requested one
    """

    def _read_table_file(self, table_file, table=None):
        for dim in ('zlevel', ):
            coord = CoordinateInfo(dim)
            coord.generic_level = True
            coord.axis = 'Z'
            self.coords[dim] = coord
        super()._read_table_file(table_file, table)

    def _read_coordinate(self, value):
        coord = super()._read_coordinate(value)
        if not coord.out_name:
            coord.out_name = coord.name
            coord.var_name = coord.name
        return coord

    def _read_variable(self, short_name, frequency):
        var = super()._read_variable(short_name, frequency)
        var.frequency = None
        var.modeling_realm = None
        return var


class CustomInfo(CMIP5Info):
    """Class to read custom var info for ESMVal.

    Parameters
    ----------
    cmor_tables_path:
        Full path to the table or name for the table if it is present in
        ESMValTool repository. If ``None``, use default tables from
        `esmvalcore/cmor/tables/custom`.

    """

    def __init__(self, cmor_tables_path: Optional[str | Path] = None) -> None:
        """Initialize class member."""
        self.coords = {}
        self.tables = {}
        self.var_to_freq: dict[str, dict] = {}
        table = TableInfo()
        table.name = 'custom'
        self.tables[table.name] = table

        # First, read default custom tables from repository
        self._cmor_folder = self._get_cmor_path('custom')
        self._read_table_dir(self._cmor_folder)

        # Second, if given, update default tables with user-defined custom
        # tables
        if cmor_tables_path is not None:
            self._user_table_folder = self._get_cmor_path(cmor_tables_path)
            if not os.path.isdir(self._user_table_folder):
                raise ValueError(
                    f"Custom CMOR tables path {self._user_table_folder} is "
                    f"not a directory"
                )
            self._read_table_dir(self._user_table_folder)
        else:
            self._user_table_folder = None

    def _read_table_dir(self, table_dir: str) -> None:
        """Read CMOR tables from directory."""
        # If present, read coordinates
        coordinates_file = os.path.join(table_dir, 'CMOR_coordinates.dat')
        if os.path.isfile(coordinates_file):
            self._read_table_file(coordinates_file)

        # Read other variables
        for dat_file in glob.glob(os.path.join(table_dir, '*.dat')):
            if dat_file == coordinates_file:
                continue
            try:
                self._read_table_file(dat_file)
            except Exception:
                msg = f"Exception raised when loading {dat_file}"
                # Logger may not be ready at this stage
                if logger.handlers:
                    logger.error(msg)
                else:
                    print(msg)
                raise

    def get_variable(
        self,
        table: str,
        short_name: str,
        derived: bool = False
    ) -> VariableInfo | None:
        """Search and return the variable info.

        Parameters
        ----------
        table:
            Table name. Ignored for custom tables.
        short_name:
            Variable's short name.
        derived:
            Variable is derived. Info retrieval for derived variables always
            looks on the default tables if variable is not found in the
            requested table. Ignored for custom tables.

        Returns
        -------
        VariableInfo | None
            `VariableInfo` object for the requested variable if found, returns
            None if not.

        """
        return self.tables['custom'].get(short_name, None)

    def _read_table_file(
        self,
        table_file: str,
        _: Optional[TableInfo] = None,
    ) -> None:
        """Read a single table file."""
        with open(table_file, 'r', encoding='utf-8') as self._current_table:
            self._read_line()
            while True:
                key, value = self._last_line_read
                if key == 'generic_levels':
                    for dim in value.split(' '):
                        coord = CoordinateInfo(dim)
                        coord.generic_level = True
                        coord.axis = 'Z'
                        self.coords[dim] = coord
                elif key == 'axis_entry':
                    self.coords[value] = self._read_coordinate(value)
                    continue
                elif key == 'variable_entry':
                    self.tables['custom'][value] = self._read_variable(
                        value, ''
                    )
                    continue
                if not self._read_line():
                    return


# Load the default tables on initializing the module.
read_cmor_tables()
