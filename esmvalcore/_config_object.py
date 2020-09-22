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


def _read_developer_config_file(filename):
    if not filename:
        filename = DEVELOPER_CONFIG

    mapping = read_config_file(filename)
    return mapping


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


def _load_default_data_reference_syntax(filename):
    drs = yaml.safe_load(open(filename, 'r'))

    for key, value in drs.items():
        drs[key] = BaseDRS(value)

    return drs


def _load_default_config(filename):
    mapping = read_config_file(filename)
    # mapping = flatten(mapping)

    global config_default

    config_default.update(mapping)

    # _load_data_reference_syntax(config_default)


def _load_user_config(filename):
    mapping = read_config_file(filename)
    # mapping = flatten(mapping)

    global config
    global config_orig

    config.clear()
    config.update(config_default)
    config.update(mapping)

    # _load_data_reference_syntax(config)

    config_orig = Config(config.copy())


# initialize default data reference syntax
DEFAULT_DRS = Path(__file__).with_name('data_reference_syntax.yml')
default_drs = _load_default_data_reference_syntax(DEFAULT_DRS)

DEVELOPER_CONFIG = Path(__file__).with_name('config-developer.yml')
DEFAULT_CONFIG = Path(__file__).with_name('config-user.yml')
USER_CONFIG = Path.home() / '.esmvaltool' / 'config-user.yml'

# initialize placeholders
config_default = Config()
config = Config()
config_orig = Config()

# update config objects
_load_default_config(DEFAULT_CONFIG)
_load_user_config(USER_CONFIG)

# TODO: load custom data_reference_syntax
# in yaml:
# data_reference_syntax:
#   CMIP6:
#     rootpath: asdf
#   CMIP5:
#     rootpath: fasf

# TODO:
#   organize files in separate config folder
#   DRS object to own file
