"""Config validation objects."""

import pprint
import re
from collections.abc import MutableMapping

from .._exceptions import SuppressedError


class InvalidConfigParameter(SuppressedError):
    """Config parameter is invalid."""


# The code for this class was take from matplotlib (v3.3) and modified to
# fit the needs of ESMValCore. Matplotlib is licenced under the terms of
# the the 'Python Software Foundation License'
# (https://www.python.org/psf/license)
class ValidatedConfig(MutableMapping, dict):
    """Based on `matplotlib.rcParams`."""

    validate = {}

    # validate values on the way in
    def __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)

    def __setitem__(self, key, val):
        try:
            cval = self.validate[key](val)
        except ValueError as verr:
            raise InvalidConfigParameter(f"Key `{key}`: {verr}") from None
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

    def __delitem__(self, key):
        dict.__delitem__(self, key)

    def find_all(self, pattern):
        """Return the subset of this Config dictionary whose keys match.

        Uses `re.search` with the given `pattern`.

        Changes to the returned dictionary are *not* propagated to the
        parent Config dictionary.
        """
        pattern_re = re.compile(pattern)
        return self.__class__((key, value) for key, value in self.items()
                              if pattern_re.search(key))

    def copy(self):
        """Copy the keys this object to a dict."""
        return {k: dict.__getitem__(self, k) for k in self}

    def clear(self):
        """Clear Config dictionary."""
        dict.clear(self)
