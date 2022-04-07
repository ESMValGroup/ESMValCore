"""Config validation objects."""

import pprint
import warnings
from collections.abc import MutableMapping
from typing import Callable, Dict, Tuple

from .._exceptions import SuppressedError
from ._config_validators import ValidationError


class InvalidConfigParameter(SuppressedError):
    """Config parameter is invalid."""


class MissingConfigParameter(UserWarning):
    """Config parameter is missing."""


# The code for this class was take from matplotlib (v3.3) and modified to
# fit the needs of ESMValCore. Matplotlib is licenced under the terms of
# the the 'Python Software Foundation License'
# (https://www.python.org/psf/license)
class ValidatedConfig(MutableMapping):
    """Based on `matplotlib.rcParams`."""

    _validate: Dict[str, Callable] = {}
    _warn_if_missing: Tuple[Tuple[str, str], ...] = ()

    # validate values on the way in
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._mapping = {}
        self.update(*args, **kwargs)

    def __setitem__(self, key, val):
        """Map key to value."""
        try:
            cval = self._validate[key](val)
        except ValidationError as verr:
            raise InvalidConfigParameter(f"Key `{key}`: {verr}") from None
        except KeyError:
            raise InvalidConfigParameter(
                f"`{key}` is not a valid config parameter.") from None

        self._mapping[key] = cval

    def __getitem__(self, key):
        """Return value mapped by key."""
        return self._mapping[key]

    def __repr__(self):
        """Return canonical string representation."""
        class_name = self.__class__.__name__
        indent = len(class_name) + 1
        repr_split = pprint.pformat(self._mapping, indent=1,
                                    width=80 - indent).split('\n')
        repr_indented = ('\n' + ' ' * indent).join(repr_split)
        return '{}({})'.format(class_name, repr_indented)

    def __str__(self):
        """Return string representation."""
        return '\n'.join(
            map('{0[0]}: {0[1]}'.format, sorted(self._mapping.items())))

    def __iter__(self):
        """Yield sorted list of keys."""
        yield from sorted(self._mapping)

    def __len__(self):
        """Return number of config keys."""
        return len(self._mapping)

    def __delitem__(self, key):
        """Delete key/value from config."""
        del self._mapping[key]

    def check_missing(self):
        """Check and warn for missing variables."""
        for (key, more_info) in self._warn_if_missing:
            if key not in self:
                more_info = f' ({more_info})' if more_info else ''
                warnings.warn(f'`{key}` is not defined{more_info}',
                              MissingConfigParameter)

    def copy(self):
        """Copy the keys/values of this object to a dict."""
        return {k: self._mapping[k] for k in self}

    def clear(self):
        """Clear Config."""
        self._mapping.clear(self)
