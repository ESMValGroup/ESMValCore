"""List of config validators."""

import warnings
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

from esmvalcore import __version__ as current_version
from esmvalcore._config import load_config_developer
from esmvalcore._recipe import TASKSEP
from esmvalcore.cmor.check import CheckLevels


class ValidationError(ValueError):
    """Custom validation error."""


# Custom warning, because DeprecationWarning is hidden by default
class ESMValToolDeprecationWarning(UserWarning):
    """Configuration key has been deprecated."""


# The code for this function was taken from matplotlib (v3.3) and modified
# to fit the needs of ESMValCore. Matplotlib is licenced under the terms of
# the the 'Python Software Foundation License'
# (https://www.python.org/psf/license)
def _make_type_validator(cls, *, allow_none=False):
    """Construct a type validator for `cls`.

    Return a validator that converts inputs to *cls* or raises (and
    possibly allows ``None`` as well).
    """
    def validator(inp):
        looks_like_none = isinstance(inp, str) and (inp.lower() == "none")
        if (allow_none and (inp is None or looks_like_none)):
            return None
        try:
            return cls(inp)
        except ValueError as err:
            if isinstance(cls, type):
                raise ValidationError(
                    f'Could not convert {repr(inp)} to {cls.__name__}'
                ) from err
            raise

    validator.__name__ = f"validate_{cls.__name__}"
    if allow_none:
        validator.__name__ += "_or_None"
    validator.__qualname__ = (validator.__qualname__.rsplit(".", 1)[0] + "." +
                              validator.__name__)
    return validator


# The code for this function was taken from matplotlib (v3.3) and modified
# to fit the needs of ESMValCore. Matplotlib is licenced under the terms of
# the the 'Python Software Foundation License'
# (https://www.python.org/psf/license)
@lru_cache()
def _listify_validator(scalar_validator,
                       allow_stringlist=False,
                       *,
                       n_items=None,
                       docstring=None,
                       return_type=list):
    """Apply the validator to a list."""
    def func(inp):
        if isinstance(inp, str):
            try:
                inp = return_type(
                    scalar_validator(val.strip()) for val in inp.split(',')
                    if val.strip())
            except Exception:
                if allow_stringlist:
                    # Sometimes, a list of colors might be a single string
                    # of single-letter colornames. So give that a shot.
                    inp = return_type(
                        scalar_validator(val.strip()) for val in inp
                        if val.strip())
                else:
                    raise
        # Allow any ordered sequence type -- generators, np.ndarray, pd.Series
        # -- but not sets, whose iteration order is non-deterministic.
        elif isinstance(inp,
                        Iterable) and not isinstance(inp, (set, frozenset)):
            # The condition on this list comprehension will preserve the
            # behavior of filtering out any empty strings (behavior was
            # from the original validate_stringlist()), while allowing
            # any non-string/text scalar values such as numbers and arrays.
            inp = return_type(
                scalar_validator(val) for val in inp
                if not isinstance(val, str) or val)
        else:
            raise ValidationError(
                f"Expected str or other non-set iterable, but got {inp}")
        if n_items is not None and len(inp) != n_items:
            raise ValidationError(f"Expected {n_items} values, "
                                  f"but there are {len(inp)} values in {inp}")
        return inp

    try:
        func.__name__ = "{}list".format(scalar_validator.__name__)
    except AttributeError:  # class instance.
        func.__name__ = "{}List".format(type(scalar_validator).__name__)
    func.__qualname__ = func.__qualname__.rsplit(".",
                                                 1)[0] + "." + func.__name__
    if docstring is not None:
        docstring = scalar_validator.__doc__
    func.__doc__ = docstring
    return func


def validate_bool(value, allow_none=False):
    """Check if the value can be evaluate as a boolean."""
    if (value is None) and allow_none:
        return value
    if not isinstance(value, bool):
        raise ValidationError(f"Could not convert `{value}` to `bool`")
    return value


def validate_path(value, allow_none=False):
    """Return a `Path` object."""
    if (value is None) and allow_none:
        return value
    try:
        path = Path(value).expanduser().absolute()
    except TypeError as err:
        raise ValidationError(f"Expected a path, but got {value}") from err
    else:
        return path


def validate_positive(value):
    """Check if number is positive."""
    if value is not None and value <= 0:
        raise ValidationError(f'Expected a positive number, but got {value}')
    return value


def _chain_validator(*funcs):
    """Chain a series of validators."""
    def chained(value):
        for func in funcs:
            value = func(value)
        return value

    return chained


validate_string = _make_type_validator(str)
validate_string_or_none = _make_type_validator(str, allow_none=True)
validate_stringlist = _listify_validator(validate_string,
                                         docstring='Return a list of strings.')
validate_int = _make_type_validator(int)
validate_int_or_none = _make_type_validator(int, allow_none=True)
validate_float = _make_type_validator(float)
validate_floatlist = _listify_validator(validate_float,
                                        docstring='Return a list of floats.')

validate_dict = _make_type_validator(dict)

validate_path_or_none = _make_type_validator(validate_path, allow_none=True)

validate_pathlist = _listify_validator(validate_path,
                                       docstring='Return a list of paths.')

validate_pathtuple = _listify_validator(validate_path,
                                        docstring='Return a tuple of paths.',
                                        return_type=tuple)

validate_int_positive = _chain_validator(validate_int, validate_positive)
validate_int_positive_or_none = _make_type_validator(validate_int_positive,
                                                     allow_none=True)


def validate_oldstyle_rootpath(value):
    """Validate `rootpath` mapping."""
    mapping = validate_dict(value)
    new_mapping = {}
    for key, paths in mapping.items():
        new_mapping[key] = validate_pathlist(paths)
    return new_mapping


def validate_oldstyle_drs(value):
    """Validate `drs` mapping."""
    mapping = validate_dict(value)
    return mapping


def validate_config_developer(value):
    """Validate and load config developer path."""
    path = validate_path_or_none(value)

    load_config_developer(path)

    return path


def validate_check_level(value):
    """Validate CMOR level check."""
    if isinstance(value, str):
        try:
            value = CheckLevels[value.upper()]
        except KeyError:
            raise ValidationError(
                f'`{value}` is not a valid strictness level') from None

    else:
        value = CheckLevels(value)

    return value


def validate_diagnostics(diagnostics):
    """Validate diagnostic location."""
    if isinstance(diagnostics, str):
        diagnostics = diagnostics.strip().split(' ')
    return {
        pattern if TASKSEP in pattern else pattern + TASKSEP + '*'
        for pattern in diagnostics or ()
    }


def deprecate(func, variable, version: str = None):
    """Wrap function to mark variables to be deprecated.

    This will give a warning if the function will be/has been deprecated.

    Parameters
    ----------
    func:
        Validator function to wrap
    variable: str
        Name of the variable to deprecate
    version: str
        Version to deprecate the variable in, should be something
        like '2.2.3'
    """
    if not version:
        version = 'a future version'

    if current_version >= version:
        warnings.warn(f"`{variable}` has been removed in {version}",
                      ESMValToolDeprecationWarning)
    else:
        warnings.warn(f"`{variable}` will be removed in {version}.",
                      ESMValToolDeprecationWarning,
                      stacklevel=2)

    return func


_validators = {
    # From user config
    'log_level': validate_string,
    'exit_on_warning': validate_bool,
    'output_dir': validate_path,
    'download_dir': validate_path,
    'auxiliary_data_dir': validate_path,
    'extra_facets_dir': validate_pathtuple,
    'compress_netcdf': validate_bool,
    'save_intermediary_cubes': validate_bool,
    'remove_preproc_dir': validate_bool,
    'max_parallel_tasks': validate_int_or_none,
    'config_developer_file': validate_config_developer,
    'profile_diagnostic': validate_bool,
    'run_diagnostic': validate_bool,
    'output_file_type': validate_string,

    # From CLI
    "resume_from": validate_pathlist,
    "skip_nonexistent": validate_bool,
    "diagnostics": validate_diagnostics,
    "check_level": validate_check_level,
    "offline": validate_bool,
    'max_years': validate_int_positive_or_none,
    'max_datasets': validate_int_positive_or_none,

    # From recipe
    'write_ncl_interface': validate_bool,

    # oldstyle
    'rootpath': validate_oldstyle_rootpath,
    'drs': validate_oldstyle_drs,

    # config location
    'config_file': validate_path,
}
