"""Config validation objects."""
from __future__ import annotations

import pprint
import warnings
from collections.abc import Callable, MutableMapping
from typing import Any

from esmvalcore.exceptions import (
    InvalidConfigParameter,
    MissingConfigParameter,
)

from ._config_validators import ValidationError


# The code for this class was take from matplotlib (v3.3) and modified to
# fit the needs of ESMValCore. Matplotlib is licenced under the terms of
# the the 'Python Software Foundation License'
# (https://www.python.org/psf/license)
class ValidatedConfig(MutableMapping):
    """Based on `matplotlib.rcParams`."""

    _validate: dict[str, Callable] = {}
    """Validation function for each configuration option.

    Each key and value in the dictionary corresponds to a configuration option
    and a corresponding validation function, respectively. Validation functions
    should have signatures ``f(value) -> validated_value`` and raise a
    ``ValidationError`` if invalid values are given.
    """

    _deprecate: dict[str, Callable] = {}
    """Handle deprecated options.

    Each key and value in the dictionary corresponds to a configuration option
    and a corresponding deprecation function, respectively. Deprecation
    functions should have signatures
    ``f(self, value, validated_value) -> None``.
    """

    _deprecated_defaults: dict[str, Any] = {}
    """Default values for deprecated options.

    Default values for deprecated options that are used if the option is not
    present in the ``_mapping`` dictionary.
    """

    _warn_if_missing: tuple[tuple[str, str], ...] = ()
    """Handle missing options.

    Each sub-tuple in the tuple consists of an option for which a warning is
    emitted and a string with more information for the user on that option.
    """

    # validate values on the way in
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._mapping = {}
        self.update(*args, **kwargs)

    def __setitem__(self, key, val):
        """Map key to value."""
        if key not in self._validate:
            raise InvalidConfigParameter(
                f"`{key}` is not a valid config parameter."
            )
        try:
            cval = self._validate[key](val)
        except ValidationError as verr:
            raise InvalidConfigParameter(f"Key `{key}`: {verr}") from None

        if key in self._deprecate:
            self._deprecate[key](self, val, cval)

        self._mapping[key] = cval

    def __getitem__(self, key):
        """Return value mapped by key."""
        try:
            return self._mapping[key]
        except KeyError:
            if key in self._deprecated_defaults:
                return self._deprecated_defaults[key]
            raise

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
        self._mapping.clear()
