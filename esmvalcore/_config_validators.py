from functools import partial
from pathlib import Path


def validate_bool(value):
    if not isinstance(value, bool):
        raise ValueError(f"Could not convert `{value}` to `bool`")
    return value


def validate_int(value, none_ok=False):
    if (value is None) and none_ok:
        return value
    if not isinstance(value, int):
        raise ValueError(f"Could not convert `{value}` to `int`")
    return value


def validate_path(value, none_ok=True):
    if (value is None) and none_ok:
        return value
    path = Path(value).expanduser().absolute()
    # it may be worth checking if a path exists for some cases
    # if not path.exists():
    #     raise ValueError(f'`{path}` does not exist')
    return path


def validate_path_list(value):
    if isinstance(value, (str, Path)):
        return validate_path(value)

    value = [validate_path(p) for p in value]
    return value


def validate_str(value, none_ok=False):
    if (value is None) and none_ok:
        return value
    if not isinstance(value, (str, bytes)):
        raise ValueError(f"Could not convert `{value}` to `str`")
    return value


def validate_str_list(value, none_ok=False):
    if isinstance(value, (str, Path)):
        return validate_path(value)

    value = [validate_str(s) for s in value]
    return value


def validate_warn(value):
    print(f'Value `{value}` not validated!')
    return value


def validate_positive(value):
    if value is not None and value < 1:
        raise ValueError(f"Must be larger be larger than 0. Got {value}.")
    return value


_validators = {
    'write_plots': validate_bool,
    'write_netcdf': validate_bool,
    'log_level': validate_str,
    'exit_on_warning': validate_bool,
    'output_file_type': validate_str,
    'output_dir': validate_path,
    'auxiliary_data_dir': validate_path,
    'compress_netcdf': validate_bool,
    'save_intermediary_cubes': validate_bool,
    'remove_preproc_dir': validate_bool,
    'max_parallel_tasks': partial(validate_int, none_ok=True),
    'config_developer_file': partial(validate_path, none_ok=True),
    'profile_diagnostic': validate_bool,
    'rootpath.CMIP5': validate_path_list,
    'rootpath.CORDEX': validate_path_list,
    'rootpath.OBS': validate_path_list,
    'rootpath.RAWOBS': validate_path_list,
    'rootpath.native6': validate_path_list,
    'rootpath.obs6': validate_path_list,
    'rootpath.default': validate_path_list,
    'drs.CMIP5': validate_str,
    'drs.CORDEX': validate_str,
    'drs.OBS': validate_str,

    # From CLI
    "skip-nonexistent": validate_bool,
    "diagnostics": validate_str_list,
    "check_level": validate_int,
    "synda_download": validate_bool,
    'max_years': validate_positive,
    'max_datasets': validate_positive,
}
