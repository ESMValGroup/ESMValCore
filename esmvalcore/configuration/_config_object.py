import pprint
import re
from collections.abc import MutableMapping
from pathlib import Path

import yaml

from .._exceptions import SuppressedError
from ._config_validators import _drs_validators, _validators


class InvalidConfigParameter(SuppressedError):
    pass


def read_config_file(config_file, folder_name=None):
    """Read config user file and store settings in a dictionary."""
    config_file = Path(config_file)
    if not config_file.exists():
        raise IOError(f'Config file `{config_file}` does not exist.')

    with open(config_file, 'r') as file:
        cfg = yaml.safe_load(file)

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
            raise InvalidConfigParameter(f"Key `{key}`: {ve}") from None
        except KeyError:
            raise InvalidConfigParameter(
                f"`{key}` is not a valid config parameter.") from None

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

    @staticmethod
    def load_from_file(filename):
        """Reload user configuration from the given file."""
        path = Path(filename).expanduser()
        if not path.exists():
            try_path = USER_CONFIG_DIR / filename
            if try_path.exists():
                path = try_path
            else:
                raise FileNotFoundError(f'No such file: `{filename}`')

        _load_user_config(path)


class BaseDRS(Config):
    validate = _drs_validators


def _load_default_config(filename):
    mapping = read_config_file(filename)

    global config_default

    config_default.update(mapping)


def _load_user_config(filename, raise_exception=True):
    try:
        mapping = read_config_file(filename)
    except IOError:
        if raise_exception:
            raise
        mapping = {}

    global config
    global config_orig

    config.clear()
    config.update(config_default)
    config.update(mapping)

    config_orig = Config(config.copy())


USER_CONFIG_DIR = Path.home() / '.esmvaltool'
DEFAULT_CONFIG = Path(__file__).with_name('config-default.yml')
USER_CONFIG = USER_CONFIG_DIR / 'config-user.yml'

# initialize placeholders
config_default = Config()
config = Config()
config_orig = Config()

# update config objects
_load_default_config(DEFAULT_CONFIG)
_load_user_config(USER_CONFIG, raise_exception=False)
