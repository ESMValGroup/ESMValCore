"""
CMOR information reader for ESMValTool.

Read variable information from CMOR 2 and CMOR 3 tables and make it easily
available for the other components of ESMValTool
"""
from functools import total_ordering
import copy
import errno
import glob
import json
import logging
import os

logger = logging.getLogger(__name__)

CMOR_TABLES = {}
"""dict of str, obj: CMOR info objects."""


def read_cmor_tables(cfg_developer):
    """Read cmor tables required in the configuration.

    Parameters
    ----------
    cfg_developer : dict of str
        Parsed config-developer file

    """
    custom = CustomInfo()
    CMOR_TABLES['custom'] = custom

    for table in cfg_developer:
        project = cfg_developer[table]

        cmor_type = project.get('cmor_type', 'CMIP5')
        table_path = project.get('cmor_path', cmor_type.lower())
        table_path = os.path.expandvars(os.path.expanduser(table_path))
        cmor_strict = project.get('cmor_strict', True)
        default_table_prefix = project.get('cmor_default_table_prefix', '')

        if cmor_type == 'CMIP5':
            CMOR_TABLES[table] = CMIP5Info(
                table_path, default=custom, strict=cmor_strict,
            )
        elif cmor_type == 'CMIP6':
            CMOR_TABLES[table] = CMIP6Info(
                table_path, default=custom, strict=cmor_strict,
                default_table_prefix=default_table_prefix
            )


class CMIP6Info(object):
    """
    Class to read CMIP6-like data request.

    This uses CMOR 3 json format

    Parameters
    ----------
    cmor_tables_path: basestring
        Path to the folder containing the Tables folder with the json files

    default: object
        Default table to look variables on if not found

    strict: bool
        If False, will look for a variable in other tables if it can not be
        found in the requested one

    """

    _CMIP_5to6_varname = {
        'sic': 'siconc',
        'sit': 'sivol',
        'tro3': 'o3',
        'usi': 'siu',
        'vsi': 'siv',
    }

    def __init__(self, cmor_tables_path, default=None, strict=True,
                 default_table_prefix=''):
        cmor_tables_path = self._get_cmor_path(cmor_tables_path)

        self._cmor_folder = os.path.join(cmor_tables_path, 'Tables')
        if glob.glob(os.path.join(self._cmor_folder, '*_CV.json')):
            self._load_controlled_vocabulary()
        self.default = default
        self.strict = strict
        self.default_table_prefix = default_table_prefix

        self.tables = {}
        self.var_to_freq = {}
        self.strict = strict

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
            'CMOR tables not found in {}'.format(cmor_tables_path)
        )

    def _load_table(self, json_file):
        with open(json_file) as inf:
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
                from collections import Counter
                var_freqs = (var.frequency for var in table.values())
                table_freq, _ = Counter(var_freqs).most_common(1)[0]
                table.frequency = table_freq
            self.tables[table.name] = table

    def _assign_dimensions(self, var, generic_levels):
        for dimension in var.dimensions:
            if dimension in generic_levels:
                coord = CoordinateInfo(dimension)
                coord.generic_level = True
                coord.axis = 'Z'
            else:
                try:
                    coord = self.coords[dimension]
                except KeyError:
                    logger.exception(
                        'Can not find dimension %s for variable %s',
                        dimension, var
                    )
                    raise

            axis = coord.axis
            if not axis:
                axis = 'none'

            var.coordinates[axis] = coord

    def _load_coordinates(self):
        self.coords = {}
        for json_file in glob.glob(
                os.path.join(self._cmor_folder, '*coordinate*.json')):
            with open(json_file) as inf:
                table_data = json.loads(inf.read())
                for coord_name in table_data['axis_entry'].keys():
                    coord = CoordinateInfo(coord_name)
                    coord.read_json(table_data['axis_entry'][coord_name])
                    self.coords[coord_name] = coord

    def _load_controlled_vocabulary(self):
        self.activities = {}
        self.institutes = {}
        for json_file in glob.glob(
                os.path.join(self._cmor_folder, '*_CV.json')):
            with open(json_file) as inf:
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
        """
        Search and return the table info.

        Parameters
        ----------
        table: basestring
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

    def get_variable(self, table_name, short_name, derived=False):
        """
        Search and return the variable info.

        Parameters
        ----------
        table_name: basestring
            Table name
        short_name: basestring
            Variable's short name
        derived: bool, optional
            Variable is derived. Info retrieval is less strict

        Returns
        -------
        VariableInfo
            Return the VariableInfo object for the requested variable if
            found, returns None if not

        """
        table = self.get_table(table_name)
        if table:
            try:
                return table[short_name]
            except KeyError:
                pass

        if short_name in CMIP6Info._CMIP_5to6_varname:
            new_short_name = CMIP6Info._CMIP_5to6_varname[short_name]
            return self.get_variable(table_name, new_short_name, derived)

        var_info = None
        if not self.strict:
            for table_vars in sorted(self.tables.values()):
                if short_name in table_vars:
                    var_info = table_vars[short_name]
                    break
        if not var_info and (not self.strict or derived):
            var_info = self.default.get_variable(table_name, short_name)

        if var_info:
            mip_info = self.get_table(table_name)
            if mip_info:
                var_info = var_info.copy()
                var_info.frequency = mip_info.frequency

        return var_info

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
    """
    Base class for the info classes.

    Provides common utility methods to read json variables
    """

    def __init__(self):
        self._json_data = {}

    def _read_json_variable(self, parameter, default=''):
        """
        Read a json parameter in json_data.

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
        """
        Read a json list parameter in json_data.

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
        """
        Class to read and store variable information.

        Parameters
        ----------
        short_name: str
            variable's short name

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
        """Coordinates"""

        self._json_data = None

    def copy(self):
        """
        Return a shallow copy of VariableInfo.

        Returns
        -------
        VariableInfo
           Shallow copy of this object
        """
        return copy.copy(self)

    def read_json(self, json_data, default_freq):
        """
        Read variable information from json.

        Non-present options will be set to empty

        Parameters
        ----------
        json_data: dict
            dictionary created by the json reader containing
            variable information

        default_freq: str
            Default frequency to use if it is not defined at variable level

        """
        self._json_data = json_data

        self.standard_name = self._read_json_variable('standard_name')
        self.long_name = self._read_json_variable('long_name')
        self.units = self._read_json_variable('units')
        self.valid_min = self._read_json_variable('valid_min')
        self.valid_max = self._read_json_variable('valid_max')
        self.positive = self._read_json_variable('positive')
        self.modeling_realm = \
            self._read_json_variable('modeling_realm').split()
        self.frequency = self._read_json_variable('frequency', default_freq)

        self.dimensions = self._read_json_variable('dimensions').split()


class CoordinateInfo(JsonInfo):
    """Class to read and store coordinate information."""

    def __init__(self, name):
        """
        Class to read and store coordinate information.

        Parameters
        ----------
        name: str
            coordinate's name

        """
        super(CoordinateInfo, self).__init__()
        self.name = name
        self.generic_level = False

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

    def read_json(self, json_data):
        """
        Read coordinate information from json.

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


class CMIP5Info(object):
    """
    Class to read CMIP5-like data request.

    Parameters
    ----------
    cmor_tables_path: basestring
       Path to the folder containing the Tables folder with the json files

    default: object
        Default table to look variables on if not found

    strict: bool
        If False, will look for a variable in other tables if it can not be
        found in the requested one

    """

    def __init__(self, cmor_tables_path, default=None, strict=True):
        cmor_tables_path = self._get_cmor_path(cmor_tables_path)

        self._cmor_folder = os.path.join(cmor_tables_path, 'Tables')
        if not os.path.isdir(self._cmor_folder):
            raise OSError(errno.ENOTDIR, "CMOR tables path is not a directory",
                          self._cmor_folder)

        self.strict = strict
        self.tables = {}
        self.coords = {}
        self.default = default
        self.strict = strict
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
        with open(table_file) as self._current_table:
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
                coord.requested = value.split(' ')
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
        """
        Search and return the table info.

        Parameters
        ----------
        table: basestring
            Table name

        Returns
        -------
        TableInfo
            Return the TableInfo object for the requested table if
            found, returns None if not

        """
        return self.tables.get(table)

    def get_variable(self, table, short_name, derived=False):
        """
        Search and return the variable info.

        Parameters
        ----------
        table: basestring
            Table name
        short_name: basestring
            Variable's short name
        derived: bool, optional
            Variable is derived. Info retrieval is less strict

        Returns
        -------
        VariableInfo
            Return the VariableInfo object for the requested variable if
            found, returns None if not

        """
        var_info = self.tables.get(table, {}).get(short_name, None)
        if var_info:
            return var_info
        if not self.strict:
            for table_vars in sorted(self.tables.values()):
                if short_name in table_vars:
                    var_info = table_vars[short_name]
                    break
        if not var_info and (derived or not self.strict):
            var_info = self.default.get_variable(table, short_name)

        if var_info:
            mip_info = self.get_table(table)
            var_info.copy()
            if mip_info:
                var_info.frequency = mip_info.frequency
        return var_info


class CustomInfo(CMIP5Info):
    """
    Class to read custom var info for ESMVal.

    Parameters
    ----------
    cmor_tables_path: basestring or None
        Full path to the table or name for the table if it is present in
        ESMValTool repository

    """

    def __init__(self, cmor_tables_path=None):
        cwd = os.path.dirname(os.path.realpath(__file__))
        self._cmor_folder = os.path.join(cwd, 'tables', 'custom')
        self.tables = {}
        self.var_to_freq = {}
        table = TableInfo()
        table.name = 'custom'
        self.tables[table.name] = table
        self._coordinates_file = os.path.join(
            self._cmor_folder,
            'CMOR_coordinates.dat',
        )
        self.coords = {}
        self._read_table_file(self._coordinates_file, self.tables['custom'])
        for dat_file in glob.glob(os.path.join(self._cmor_folder, '*.dat')):
            if dat_file == self._coordinates_file:
                continue
            try:
                self._read_table_file(dat_file, self.tables['custom'])
            except Exception:
                msg = f"Exception raised when loading {dat_file}"
                # Logger may not be ready at this stage
                if logger.handlers:
                    logger.error(msg)
                else:
                    print(msg)
                raise

    def get_table(self, table):
        """
        Search and return the table info.

        Parameters
        ----------
        table: basestring
            Table name

        Returns
        -------
        TableInfo
            Return the TableInfo object for the requested table if
            found, returns None if not

        """
        return self.tables.get(table)

    def get_variable(self, table, short_name, derived=False):
        """
        Search and return the variable info.

        Parameters
        ----------
        table: basestring
            Table name
        short_name: basestring
            Variable's short name
        derived: bool, optional
            Variable is derived. Info retrieval is less strict

        Returns
        -------
        VariableInfo
            Return the VariableInfo object for the requested variable if
            found, returns None if not

        """
        return self.tables['custom'].get(short_name, None)

    def _read_table_file(self, table_file, table=None):
        with open(table_file) as self._current_table:
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
                    table[value] = self._read_variable(value, None)
                    continue
                if not self._read_line():
                    return
