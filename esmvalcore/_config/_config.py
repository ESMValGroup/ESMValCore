"""Functions dealing with config-user.yml / config-developer.yml."""
import datetime
import logging
import os
import warnings
from pathlib import Path

import yaml

from esmvalcore.cmor.table import CMOR_TABLES, read_cmor_tables

logger = logging.getLogger(__name__)

CFG = {}


def read_config_user_file(config_file, folder_name, options=None):
    """Read config user file and store settings in a dictionary."""
    if not config_file:
        config_file = '~/.esmvaltool/config-user.yml'
    config_file = os.path.abspath(
        os.path.expandvars(os.path.expanduser(config_file)))
    # Read user config file
    if not os.path.exists(config_file):
        print(f"ERROR: Config file {config_file} does not exist")

    with open(config_file, 'r') as file:
        cfg = yaml.safe_load(file)

    # DEPRECATED: remove in v2.4
    for setting in ('write_plots', 'write_netcdf'):
        if setting in cfg:
            msg = (
                f"Using '{setting}' in {config_file} is deprecated and will "
                "be removed in ESMValCore version 2.4. For diagnostics "
                "that support this setting, it should be set in the "
                "diagnostic script section of the recipe instead. "
                f"Remove the setting from {config_file} to get rid of this "
                "warning message.")
            print(f"Warning: {msg}")
            warnings.warn(DeprecationWarning(msg))

    if options is None:
        options = dict()
    for key, value in options.items():
        cfg[key] = value
        # DEPRECATED: remove in v2.4
        if key in ('write_plots', 'write_netcdf'):
            msg = (
                f"Setting '{key}' from the command line is deprecated and "
                "will be removed in ESMValCore version 2.4. For diagnostics "
                "that support this setting, it should be set in the "
                "diagnostic script section of the recipe instead.")
            print(f"Warning: {msg}")
            warnings.warn(DeprecationWarning(msg))

    # set defaults
    defaults = {
        'compress_netcdf': False,
        'exit_on_warning': False,
        'output_file_type': 'png',
        'output_dir': 'esmvaltool_output',
        'auxiliary_data_dir': 'auxiliary_data',
        'save_intermediary_cubes': False,
        'remove_preproc_dir': True,
        'max_parallel_tasks': None,
        'run_diagnostic': True,
        'profile_diagnostic': False,
        'config_developer_file': None,
        'drs': {},
        # DEPRECATED: remove default settings below in v2.4
        'write_plots': True,
        'write_netcdf': True,
    }

    for key in defaults:
        if key not in cfg:
            logger.info(
                "No %s specification in config file, "
                "defaulting to %s", key, defaults[key])
            cfg[key] = defaults[key]

    cfg['output_dir'] = _normalize_path(cfg['output_dir'])
    cfg['auxiliary_data_dir'] = _normalize_path(cfg['auxiliary_data_dir'])

    cfg['config_developer_file'] = _normalize_path(
        cfg['config_developer_file'])

    for key in cfg['rootpath']:
        root = cfg['rootpath'][key]
        if isinstance(root, str):
            cfg['rootpath'][key] = [_normalize_path(root)]
        else:
            cfg['rootpath'][key] = [_normalize_path(path) for path in root]

    # insert a directory date_time_recipe_usertag in the output paths
    now = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    new_subdir = '_'.join((folder_name, now))
    cfg['output_dir'] = os.path.join(cfg['output_dir'], new_subdir)

    # create subdirectories
    cfg['preproc_dir'] = os.path.join(cfg['output_dir'], 'preproc')
    cfg['work_dir'] = os.path.join(cfg['output_dir'], 'work')
    cfg['plot_dir'] = os.path.join(cfg['output_dir'], 'plots')
    cfg['run_dir'] = os.path.join(cfg['output_dir'], 'run')

    # Read developer configuration file
    load_config_developer(cfg['config_developer_file'])

    return cfg


def _normalize_path(path):
    """Normalize paths.

    Expand ~ character and environment variables and convert path to absolute.

    Parameters
    ----------
    path: str
        Original path

    Returns
    -------
    str:
        Normalized path
    """
    if path is None:
        return None
    return os.path.abspath(os.path.expanduser(os.path.expandvars(path)))


def read_config_developer_file(cfg_file=None):
    """Read the developer's configuration file."""
    if cfg_file is None:
        cfg_file = Path(__file__).parents[1] / 'config-developer.yml'

    with open(cfg_file, 'r') as file:
        cfg = yaml.safe_load(file)

    return cfg


def load_config_developer(cfg_file=None):
    """Load the config developer file and initialize CMOR tables."""
    cfg_developer = read_config_developer_file(cfg_file)
    for key, value in cfg_developer.items():
        CFG[key] = value
    read_cmor_tables(CFG)


def get_project_config(project):
    """Get developer-configuration for project."""
    logger.debug("Retrieving %s configuration", project)
    if project in CFG:
        return CFG[project]
    raise ValueError(f"Project '{project}' not in config-developer.yml")


def get_institutes(variable):
    """Return the institutes given the dataset name in CMIP5 and CMIP6."""
    dataset = variable['dataset']
    project = variable['project']
    logger.debug("Retrieving institutes for dataset %s", dataset)
    try:
        return CMOR_TABLES[project].institutes[dataset]
    except (KeyError, AttributeError):
        pass
    return CFG.get(project, {}).get('institutes', {}).get(dataset, [])


def get_activity(variable):
    """Return the activity given the experiment name in CMIP6."""
    project = variable['project']
    try:
        exp = variable['exp']
        logger.debug("Retrieving activity_id for experiment %s", exp)
        if isinstance(exp, list):
            return [CMOR_TABLES[project].activities[value][0] for value in exp]
        return CMOR_TABLES[project].activities[exp][0]
    except (KeyError, AttributeError):
        return None
