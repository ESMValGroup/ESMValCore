import pprint
import re
from collections.abc import MutableMapping
from pathlib import Path

from esmvalcore._config_validators import _validators


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


def _load_config_file(filename):
    from esmvalcore._config import read_config_user_file
    mapping = read_config_user_file(filename)
    return flatten(mapping)


DEFAULT_CONFIG = Path(__file__).with_name('config-user.yml')
USER_CONFIG = Path.home() / '.esmvaltool' / 'config-user.yml'

config_default = Config()
config_default.update(_load_config_file(DEFAULT_CONFIG))

config = Config()
config.update(config_default)
config.update(_load_config_file(USER_CONFIG))

config_orig = Config(config.copy())
