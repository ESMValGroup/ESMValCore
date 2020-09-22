import pprint
import re
from collections.abc import MutableMapping
from pathlib import Path

import yaml

from esmvalcore._config_validators import _drs_validators, _validators


def flatten(d, parent_key='', sep='.'):
    """Flatten nested dictionary."""
    items = []
    for key, val in d.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(val, MutableMapping):
            items.extend(flatten(val, new_key, sep=sep).items())
        else:
            items.append((new_key, val))
    return dict(items)


def read_config_file(config_file, folder_name=None):
    """Read config user file and store settings in a dictionary."""
    config_file = Path(config_file)
    if not config_file.exists():
        print(f"ERROR: Config file {config_file} does not exist")

    with open(config_file, 'r') as file:
        cfg = yaml.safe_load(file)

    # short-hand for including site-specific variables
    site = cfg.pop('site', None)
    if site:
        cfg['include'] = Path(__file__).with_name(f'config-{site}.yml')

    # include nested yaml files here to be ahead of dictionary flattening
    # this is to ensure variables are updated at root level, specifically
    # `rootpath`/`drs`
    include = cfg.pop('include', None)
    if include:
        for try_path in (
                Path(include).expanduser().absolute(),
                Path(__file__).parent / include,
        ):
            if try_path.exists():
                include = try_path
                break

        include_cfg = read_config_file(include)
        cfg.update(include_cfg)

    return cfg


class Config(MutableMapping, dict):
    """Based on `matplotlib.rcParams`."""
    validate = _validators

    # validate values on the way in
    def __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)

    def __setitem__(self, key, val):
        try:
            cval = self.validate[key](val)
        except ValueError as ve:
            raise ValueError(f"Key `{key}`: {ve}") from None
        except KeyError as ke:
            raise KeyError(f"`{key}` is not a valid config parameter.") from ke

        dict.__setitem__(self, key, cval)

    def __getitem__(self, key):
        return dict.__getitem__(self, key)

    def __repr__(self):
        class_name = self.__class__.__name__
        indent = len(class_name) + 1
        repr_split = pprint.pformat(dict(self), indent=1,
                                    width=80 - indent).split('\n')
        repr_indented = ('\n' + ' ' * indent).join(repr_split)
        return '{}({})'.format(class_name, repr_indented)

    def __str__(self):
        return '\n'.join(map('{0[0]}: {0[1]}'.format, sorted(self.items())))

    def __iter__(self):
        """Yield sorted list of keys."""
        yield from sorted(dict.__iter__(self))

    def __len__(self):
        return dict.__len__(self)

    def find_all(self, pattern):
        """Return the subset of this Config dictionary whose keys match, using
        `re.search` with the given `pattern`.

        Changes to the returned dictionary are *not* propagated to the
        parent Config dictionary.
        """
        pattern_re = re.compile(pattern)
        return Config((key, value) for key, value in self.items()
                      if pattern_re.search(key))

    def copy(self):
        return {k: dict.__getitem__(self, k) for k in self}

    def clear(self):
        """Clear Config dictionary."""
        dict.clear(self)

    def select_group(self, group):
        prefix = f'{group}.'
        subset = self.find_all(f'^{prefix}')
        return dict((key[len(prefix):], item) for key, item in subset.items())


class BaseDRS(Config):
    validate = _drs_validators

    @property
    def rootpath(self):
        rootpath = self['rootpath']
        if not rootpath:
            rootpath = config['default_inputpath']
        return rootpath


def _load_default_data_reference_syntax(filename):
    drs = yaml.safe_load(open(filename, 'r'))

    global drs_config_default

    for key, value in drs.items():
        drs_config_default[key] = BaseDRS(value)


def _load_data_reference_syntax(config):
    drs = config['data_reference_syntax']

    global drs_config
    global drs_config_orig

    for key, value in drs.items():
        project = key.split('_')[0]

        if project in drs_config_default:
            new = drs_config_default[project].copy()
            new.update(value)
        else:
            new = BaseDRS(value)

        drs_config[key] = new

    drs_config_orig = drs_config.copy()


def _load_default_config(filename):
    mapping = read_config_file(filename)

    global config_default

    config_default.update(mapping)


def _load_user_config(filename):
    mapping = read_config_file(filename)

    global config
    global config_orig

    config.clear()
    config.update(config_default)
    config.update(mapping)

    config_orig = Config(config.copy())


DEFAULT_CONFIG = Path(__file__).with_name('config-user.yml')
USER_CONFIG = Path.home() / '.esmvaltool' / 'config-user.yml'

# initialize placeholders
config_default = Config()
config = Config()
config_orig = Config()

# update config objects
_load_default_config(DEFAULT_CONFIG)
_load_user_config(USER_CONFIG)

DEFAULT_DRS = Path(__file__).with_name('data_reference_syntax.yml')

# initialize placeholders
drs_config_default = dict()
drs_config = dict()
drs_config_orig = dict()

# update data data reference syntax
_load_default_data_reference_syntax(DEFAULT_DRS)
_load_data_reference_syntax(config)

# TODO:
#   organize files in separate config folder
#   DRS object to own file
