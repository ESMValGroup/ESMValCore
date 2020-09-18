from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path


def _make_type_validator(cls, *, allow_none=False):
    """Return a validator that converts inputs to *cls* or raises (and possibly
    allows ``None`` as well)."""
    def validator(s):
        if (allow_none
                and (s is None or isinstance(s, str) and s.lower() == "none")):
            return None
        try:
            return cls(s)
        except ValueError as e:
            if isinstance(cls, type):
                raise ValueError(
                    f'Could not convert {s!r} to {cls.__name__}') from e
            else:
                raise

    validator.__name__ = f"validate_{cls.__name__}"
    if allow_none:
        validator.__name__ += "_or_None"
    validator.__qualname__ = (validator.__qualname__.rsplit(".", 1)[0] + "." +
                              validator.__name__)
    return validator


@lru_cache
def _listify_validator(scalar_validator,
                       allow_stringlist=False,
                       *,
                       n=None,
                       doc=None):
    """Apply the validator to a list."""
    def f(s):
        if isinstance(s, str):
            try:
                val = [
                    scalar_validator(v.strip()) for v in s.split(',')
                    if v.strip()
                ]
            except Exception:
                if allow_stringlist:
                    # Sometimes, a list of colors might be a single string
                    # of single-letter colornames. So give that a shot.
                    val = [scalar_validator(v.strip()) for v in s if v.strip()]
                else:
                    raise
        # Allow any ordered sequence type -- generators, np.ndarray, pd.Series
        # -- but not sets, whose iteration order is non-deterministic.
        elif isinstance(s, Iterable) and not isinstance(s, (set, frozenset)):
            # The condition on this list comprehension will preserve the
            # behavior of filtering out any empty strings (behavior was
            # from the original validate_stringlist()), while allowing
            # any non-string/text scalar values such as numbers and arrays.
            val = [
                scalar_validator(v) for v in s if not isinstance(v, str) or v
            ]
        else:
            raise ValueError(
                f"Expected str or other non-set iterable, but got {s}")
        if n is not None and len(val) != n:
            raise ValueError(
                f"Expected {n} values, but there are {len(val)} values in {s}")
        return val

    try:
        f.__name__ = "{}list".format(scalar_validator.__name__)
    except AttributeError:  # class instance.
        f.__name__ = "{}List".format(type(scalar_validator).__name__)
    f.__qualname__ = f.__qualname__.rsplit(".", 1)[0] + "." + f.__name__
    f.__doc__ = doc if doc is not None else scalar_validator.__doc__
    return f


def validate_bool(value, allow_none=False):
    """Check if the value can be evaluate as a boolean."""
    if (value is None) and allow_none:
        return value
    if not isinstance(value, bool):
        raise ValueError(f"Could not convert `{value}` to `bool`")
    return value


def validate_path(value, allow_none=False):
    """Return a path object."""
    if (value is None) and allow_none:
        return value
    try:
        path = Path(value).expanduser().absolute()
    except TypeError as e:
        raise ValueError(f"Expected a path, but got {value}") from e
    else:
        return path


def validate_positive(value):
    """Check if number is positive."""
    if value < 0:
        raise ValueError(f'Expected a positive number, but got {value}')
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
                                         doc='return a list of strings')
validate_int = _make_type_validator(int)
validate_int_or_none = _make_type_validator(int, allow_none=True)
validate_float = _make_type_validator(float)
validate_float_or_none = _make_type_validator(float, allow_none=True)
validate_floatlist = _listify_validator(validate_float,
                                        doc='return a list of floats')

validate_path_or_none = _make_type_validator(validate_path, allow_none=True)

validate_pathlist = _listify_validator(validate_path,
                                       doc='return a list of paths')

validate_int_positive = _chain_validator(validate_int, validate_positive)
validate_int_positive_or_none = _make_type_validator(validate_int_positive,
                                                     allow_none=True)


def validate_positive(value):
    """Reject negative values."""
    if value is not None and value < 1:
        raise ValueError(f"Must be larger be larger than 0. Got {value}.")
    return value


def validate_check_level(value):
    from .cmor.check import CheckLevels

    if isinstance(value, str):
        return CheckLevels[value.upper()]

    return CheckLevels(value)


def validate_diagnostics(diagnostics):
    from ._recipe import TASKSEP

    if isinstance(diagnostics, str):
        diagnostics = diagnostics.split(' ')
    return {
        pattern if TASKSEP in pattern else pattern + TASKSEP + '*'
        for pattern in diagnostics or ()
    }


_validators = {
    # From user config
    'write_plots': validate_bool,
    'write_netcdf': validate_bool,
    'log_level': validate_string,
    'exit_on_warning': validate_bool,
    'output_file_type': validate_string,
    'output_dir': validate_path,
    'auxiliary_data_dir': validate_path,
    'compress_netcdf': validate_bool,
    'save_intermediary_cubes': validate_bool,
    'remove_preproc_dir': validate_bool,
    'max_parallel_tasks': validate_int_or_none,
    'config_developer_file': validate_path_or_none,
    'profile_diagnostic': validate_bool,
    'rootpath.CMIP5': validate_pathlist,
    'rootpath.CORDEX': validate_pathlist,
    'rootpath.OBS': validate_pathlist,
    'rootpath.RAWOBS': validate_pathlist,
    'rootpath.native6': validate_pathlist,
    'rootpath.obs6': validate_pathlist,
    'rootpath.default': validate_pathlist,
    'drs.CMIP5': validate_string,
    'drs.CORDEX': validate_string,
    'drs.OBS': validate_string,

    # From CLI
    "skip-nonexistent": validate_bool,
    "diagnostics": validate_diagnostics,
    "check_level": validate_check_level,
    "synda_download": validate_bool,
    'max_years': validate_int_positive_or_none,
    'max_datasets': validate_int_positive_or_none,

    # From recipe
    'write_ncl_interface': validate_bool,
}
