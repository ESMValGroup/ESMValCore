"""List of config validators."""

from __future__ import annotations

import logging
import os.path
import warnings
from collections.abc import Callable, Iterable
from functools import lru_cache, partial
from importlib.resources import files as importlib_files
from pathlib import Path
from typing import TYPE_CHECKING, Any

from packaging import version

from esmvalcore import __version__ as current_version
from esmvalcore.cmor.check import CheckLevels
from esmvalcore.config._config import TASKSEP, load_config_developer
from esmvalcore.exceptions import (
    ESMValCoreDeprecationWarning,
    InvalidConfigParameter,
)

if TYPE_CHECKING:
    from esmvalcore.config._validated_config import ValidatedConfig

logger = logging.getLogger(__name__)


SEARCH_ESGF_OPTIONS = (
    "never",  # Never search ESGF for files
    "when_missing",  # Only search ESGF if no local files are available
    "always",  # Always search ESGF for files
)


class ValidationError(ValueError):
    """Custom validation error."""


# The code for this function was taken from matplotlib (v3.3) and modified
# to fit the needs of ESMValCore. Matplotlib is licenced under the terms of
# the the 'Python Software Foundation License'
# (https://www.python.org/psf/license)
def _make_type_validator(cls: Any, *, allow_none: bool = False) -> Any:
    """Construct a type validator for `cls`.

    Return a validator that converts inputs to *cls* or raises (and
    possibly allows ``None`` as well).
    """

    def validator(inp):
        looks_like_none = isinstance(inp, str) and (inp.lower() == "none")
        if allow_none and (inp is None or looks_like_none):
            return None
        try:
            return cls(inp)
        except (ValueError, TypeError) as err:
            if isinstance(cls, type):
                msg = f"Could not convert {inp!r} to {cls.__name__}"
                raise ValidationError(msg) from err
            raise

    validator.__name__ = f"validate_{cls.__name__}"
    if allow_none:
        validator.__name__ += "_or_None"
    validator.__qualname__ = (
        validator.__qualname__.rsplit(".", 1)[0] + "." + validator.__name__
    )
    return validator


# The code for this function was taken from matplotlib (v3.3) and modified
# to fit the needs of ESMValCore. Matplotlib is licenced under the terms of
# the the 'Python Software Foundation License'
# (https://www.python.org/psf/license)
@lru_cache
def _listify_validator(
    scalar_validator,
    allow_stringlist=False,
    *,
    n_items=None,
    docstring=None,
    return_type=list,
):
    """Apply the validator to a list."""

    def func(inp):
        if isinstance(inp, str):
            try:
                inp = return_type(
                    scalar_validator(val.strip())
                    for val in inp.split(",")
                    if val.strip()
                )
            except Exception:
                if allow_stringlist:
                    # Sometimes, a list of colors might be a single string
                    # of single-letter colornames. So give that a shot.
                    inp = return_type(
                        scalar_validator(val.strip())
                        for val in inp
                        if val.strip()
                    )
                else:
                    raise
        # Allow any ordered sequence type -- generators, np.ndarray, pd.Series
        # -- but not sets, whose iteration order is non-deterministic.
        elif isinstance(inp, Iterable) and not isinstance(
            inp,
            (set, frozenset),
        ):
            # The condition on this list comprehension will preserve the
            # behavior of filtering out any empty strings (behavior was
            # from the original validate_stringlist()), while allowing
            # any non-string/text scalar values such as numbers and arrays.
            inp = return_type(
                scalar_validator(val)
                for val in inp
                if not isinstance(val, str) or val
            )
        else:
            msg = f"Expected str or other non-set iterable, but got {inp}"
            raise ValidationError(msg)
        if n_items is not None and len(inp) != n_items:
            msg = (
                f"Expected {n_items} values, "
                f"but there are {len(inp)} values in {inp}"
            )
            raise ValidationError(msg)
        return inp

    try:
        func.__name__ = f"{scalar_validator.__name__}list"
    except AttributeError:  # class instance.
        func.__name__ = f"{type(scalar_validator).__name__}List"
    func.__qualname__ = (
        func.__qualname__.rsplit(".", 1)[0] + "." + func.__name__
    )
    if docstring is not None:
        docstring = scalar_validator.__doc__
    func.__doc__ = docstring
    return func


def validate_bool(value, allow_none=False):
    """Check if the value can be evaluate as a boolean."""
    if (value is None) and allow_none:
        return value
    if not isinstance(value, bool):
        msg = f"Could not convert `{value}` to `bool`"
        raise ValidationError(msg)
    return value


def validate_path(value, allow_none=False):
    """Return a `Path` object."""
    if (value is None) and allow_none:
        return value
    try:
        path = Path(os.path.expandvars(value)).expanduser().absolute()
    except TypeError as err:
        msg = f"Expected a path, but got {value}"
        raise ValidationError(msg) from err
    else:
        return path


def validate_positive(value):
    """Check if number is positive."""
    msg = f"Expected a positive number, but got {value}"
    try:
        if value is not None and value <= 0:
            raise ValidationError(msg)
    except TypeError as err:
        raise ValidationError(msg) from err
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
validate_stringlist = _listify_validator(
    validate_string,
    docstring="Return a list of strings.",
)

validate_bool_or_none = partial(validate_bool, allow_none=True)
validate_int = _make_type_validator(int)
validate_int_or_none = _make_type_validator(int, allow_none=True)
validate_float = _make_type_validator(float)
validate_floatlist = _listify_validator(
    validate_float,
    docstring="Return a list of floats.",
)

validate_dict = _make_type_validator(dict)

validate_path_or_none = _make_type_validator(validate_path, allow_none=True)

validate_pathlist = _listify_validator(
    validate_path,
    docstring="Return a list of paths.",
)

validate_int_positive = _chain_validator(validate_int, validate_positive)
validate_int_positive_or_none = _make_type_validator(
    validate_int_positive,
    allow_none=True,
)


def validate_rootpath(value):
    """Validate `rootpath` mapping."""
    mapping = validate_dict(value)
    new_mapping = {}
    for key, paths in mapping.items():
        if key == "obs4mips":
            logger.warning(
                "Correcting capitalization, project 'obs4mips' should be "
                "written as 'obs4MIPs' in configured 'rootpath'",
            )
            key = "obs4MIPs"  # noqa: PLW2901
        if isinstance(paths, Path):
            paths = str(paths)  # noqa: PLW2901
        if isinstance(paths, (str, list)):
            new_mapping[key] = validate_pathlist(paths)
        else:
            validate_dict(paths)

            # dask.config.merge cannot handle pathlib.Path objects as dict keys
            # -> we convert the validated Path back to a string and handle this
            # downstream in local.py (see also
            # https://github.com/ESMValGroup/ESMValCore/issues/2577)
            new_mapping[key] = {
                str(validate_path(path)): validate_string(drs)
                for path, drs in paths.items()
            }

    return new_mapping


def validate_drs(value):
    """Validate `drs` mapping."""
    mapping = validate_dict(value)
    new_mapping = {}
    for key, drs in mapping.items():
        if key == "obs4mips":
            logger.warning(
                "Correcting capitalization, project 'obs4mips' should be "
                "written as 'obs4MIPs' in configured 'drs'",
            )
            key = "obs4MIPs"  # noqa: PLW2901
        new_mapping[key] = validate_string(drs)
    return new_mapping


def validate_config_developer(value):
    """Validate and load config developer path."""
    path = validate_path_or_none(value)
    if path is None:
        path = importlib_files("esmvalcore") / "config-developer.yml"
    load_config_developer(path)

    return path


def validate_check_level(value):
    """Validate CMOR level check."""
    msg = f"`{value}` is not a valid strictness level"

    if isinstance(value, str):
        try:
            value = CheckLevels[value.upper()]
        except KeyError:
            raise ValidationError(msg) from None

    else:
        try:
            value = CheckLevels(value)
        except (ValueError, TypeError) as err:
            raise ValidationError(msg) from err

    return value


def validate_search_esgf(value):
    """Validate options for ESGF search."""
    value = validate_string(value)
    value = value.lower()
    if value not in SEARCH_ESGF_OPTIONS:
        msg = (
            f"`{value}` is not a valid option ESGF search option, possible "
            f"values are {SEARCH_ESGF_OPTIONS}"
        )
        raise ValidationError(msg) from None
    return value


def validate_diagnostics(
    diagnostics: Iterable[str] | str | None,
) -> set[str] | None:
    """Validate diagnostic location."""
    if diagnostics is None:
        return None
    if isinstance(diagnostics, str):
        diagnostics = diagnostics.strip().split(" ")
    try:
        validated_diagnostics = {
            pattern if TASKSEP in pattern else pattern + TASKSEP + "*"
            for pattern in diagnostics or ()
        }
    except TypeError as err:
        msg = (
            f"Expected str or Iterable[str] for diagnostics, got "
            f"`{diagnostics}`"
        )
        raise ValidationError(msg) from err
    return validated_diagnostics


# TODO: remove in v2.14.0
def validate_extra_facets_dir(value):
    """Validate extra_facets_dir."""
    if isinstance(value, tuple):
        msg = (
            "Specifying `extra_facets_dir` as tuple has been deprecated in "
            "ESMValCore version 2.12.0 and is scheduled for removal in "
            "version 2.14.0. Please use a list instead."
        )
        warnings.warn(msg, ESMValCoreDeprecationWarning, stacklevel=2)
        value = list(value)
    return validate_pathlist(value)


def validate_projects(value: Any) -> Any:
    """Validate projects mapping."""
    mapping = validate_dict(value)
    options_for_project: dict[str, Callable[[Any], Any]] = {
        "extra_facets": validate_dict,
    }
    for project, project_config in mapping.items():
        for option, val in project_config.items():
            if option not in options_for_project:
                msg = (
                    f"`{option}` is not a valid option for the `projects` "
                    f"configuration of project `{project}`, possible options "
                    f"are {list(options_for_project)}"
                )
                raise ValidationError(msg) from None
            mapping[project][option] = options_for_project[option](val)
    return mapping


_validators = {
    "auxiliary_data_dir": validate_path,
    "check_level": validate_check_level,
    "compress_netcdf": validate_bool,
    "config_developer_file": validate_config_developer,
    "dask": validate_dict,
    "diagnostics": validate_diagnostics,
    "download_dir": validate_path,
    "drs": validate_drs,
    "exit_on_warning": validate_bool,
    "log_level": validate_string,
    "logging": validate_dict,
    "max_datasets": validate_int_positive_or_none,
    "max_parallel_tasks": validate_int_or_none,
    "max_years": validate_int_positive_or_none,
    "output_dir": validate_path,
    "output_file_type": validate_string,
    "profile_diagnostic": validate_bool,
    "projects": validate_projects,
    "remove_preproc_dir": validate_bool,
    "resume_from": validate_pathlist,
    "rootpath": validate_rootpath,
    "run_diagnostic": validate_bool,
    "save_intermediary_cubes": validate_bool,
    "search_esgf": validate_search_esgf,
    "skip_nonexistent": validate_bool,
    # From recipe
    "write_ncl_interface": validate_bool,
    # TODO: remove in v2.14.0
    # config location
    "config_file": validate_path,
    # TODO: remove in v2.15.0
    "extra_facets_dir": validate_extra_facets_dir,
}


# Handle deprecations (using ``ValidatedConfig._deprecate``)


def _handle_deprecation(
    option: str,
    deprecated_version: str,
    remove_version: str,
    more_info: str,
) -> None:
    """Handle deprecated configuration option."""
    if version.parse(current_version) >= version.parse(remove_version):
        remove_msg = (
            f"The configuration option or command line argument `{option}` "
            f"has been removed in ESMValCore version {remove_version}."
            f"{more_info}"
        )
        raise InvalidConfigParameter(remove_msg)

    deprecation_msg = (
        f"The configuration option or command line argument `{option}` has "
        f"been deprecated in ESMValCore version {deprecated_version} and is "
        f"scheduled for removal in version {remove_version}.{more_info}"
    )
    warnings.warn(deprecation_msg, ESMValCoreDeprecationWarning, stacklevel=2)


# TODO: remove in v2.14.0
def deprecate_config_file(
    validated_config: ValidatedConfig,
    value: Any,
    validated_value: Any,
) -> None:
    """Deprecate ``config_file`` option.

    Parameters
    ----------
    validated_config:
        ``ValidatedConfig`` instance which will be modified in place.
    value:
        Raw input value for ``config_file`` option.
    validated_value:
        Validated value for ``config_file`` option.

    """
    validated_config  # noqa: B018
    value  # noqa: B018
    validated_value  # noqa: B018
    option = "config_file"
    deprecated_version = "2.12.0"
    remove_version = "2.14.0"
    more_info = " Please use the option `config_dir` instead."
    _handle_deprecation(option, deprecated_version, remove_version, more_info)


# TODO: remove in v2.15.0
def deprecate_extra_facets_dir(
    validated_config: ValidatedConfig,
    value: Any,
    validated_value: Any,
) -> None:
    """Deprecate ``extra_facets_dir`` option.

    Parameters
    ----------
    validated_config:
        ``ValidatedConfig`` instance which will be modified in place.
    value:
        Raw input value for ``extra_facets_dir`` option.
    validated_value:
        Validated value for ``extra_facets_dir`` option.

    """
    validated_config  # noqa: B018
    value  # noqa: B018
    validated_value  # noqa: B018
    option = "extra_facets_dir"
    deprecated_version = "2.13.0"
    remove_version = "2.15.0"
    more_info = (
        " Please use the option `extra_facets` instead (see "
        "https://github.com/ESMValGroup/ESMValCore/pull/2747 for details)."
    )
    if os.environ.get("ESMVALTOOL_USE_NEW_EXTRA_FACETS_CONFIG"):
        more_info += (
            " Since the environment variable "
            "ESMVALTOOL_USE_NEW_EXTRA_FACETS_CONFIG is set, "
            "`extra_facets_dir` is ignored. To silent this warning, omit "
            "`extra_facets_dir`."
        )
    _handle_deprecation(option, deprecated_version, remove_version, more_info)


# Example usage: see removed files in
# https://github.com/ESMValGroup/ESMValCore/pull/2213
_deprecators: dict[str, Callable] = {
    "config_file": deprecate_config_file,  # TODO: remove in v2.14.0
    "extra_facets_dir": deprecate_extra_facets_dir,  # TODO: remove in v2.15.0
}


# Default values for deprecated options
# Example usage: see removed files in
# https://github.com/ESMValGroup/ESMValCore/pull/2213
_deprecated_options_defaults: dict[str, Any] = {
    "config_file": None,  # TODO: remove in v2.14.0
    "extra_facets_dir": [],  # TODO: remove in v2.15.0
}
