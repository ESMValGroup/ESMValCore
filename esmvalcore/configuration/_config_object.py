import os
from pathlib import Path

import yaml

from ._config_validators import _validators
from ._validated_config import ValidatedConfig


class ESMValCoreConfig(ValidatedConfig):
    """The ESMValCore config object."""
    validate = _validators

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


def read_config_file(config_file):
    """Read config user file and store settings in a dictionary."""
    config_file = Path(config_file)
    if not config_file.exists():
        raise IOError(f'Config file `{config_file}` does not exist.')

    with open(config_file, 'r') as file:
        cfg = yaml.safe_load(file)

    return cfg


def _load_default_config(filename: str, drs_filename: str = None):
    """Load the default configuration."""
    mapping = read_config_file(filename)

    global config_default

    config_default.update(mapping)

    if drs_filename:
        drs_mapping = read_config_file(drs_filename)
        config_default.update(drs_mapping)


def _load_user_config(filename: str, raise_exception: bool = True):
    """Load user configuration from the given file (`filename`).

    The config cleared and updated in-place.

    Parameters
    ----------
    raise_exception : bool
        Raise an exception if `filename` can not be found (default).
        Otherwise, silently pass and use the default configuration. This
        setting is necessary for the case where `.esmvalcore/config-user.yml`
        has not been defined (i.e. first start).
    """
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

    config_orig = ESMValCoreConfig(config.copy())


def get_user_config_location():
    """Check if environment variable `ESMVALTOOL_CONFIG` exists, otherwise use
    the default config location."""
    try:
        config_location = Path(os.environ['ESMVALTOOL_CONFIG'])
    except KeyError:
        config_location = USER_CONFIG_DIR / 'config-user.yml'

    return config_location


DEFAULT_CONFIG_DIR = Path(__file__).parent
DEFAULT_CONFIG = DEFAULT_CONFIG_DIR / 'config-default.yml'
DEFAULT_DRS = DEFAULT_CONFIG_DIR / 'drs-default.yml'

USER_CONFIG_DIR = Path.home() / '.esmvaltool'
USER_CONFIG = get_user_config_location()

# initialize placeholders
config_default = ESMValCoreConfig()
config = ESMValCoreConfig()
config_orig = ESMValCoreConfig()

# update config objects
_load_default_config(DEFAULT_CONFIG, DEFAULT_DRS)
_load_user_config(USER_CONFIG, raise_exception=False)
