from pathlib import Path

import yaml

from ._config_validators import _validators
from ._validated_config import ValidatedConfig


def read_config_file(config_file, folder_name=None):
    """Read config user file and store settings in a dictionary."""
    config_file = Path(config_file)
    if not config_file.exists():
        raise IOError(f'Config file `{config_file}` does not exist.')

    with open(config_file, 'r') as file:
        cfg = yaml.safe_load(file)

    return cfg


class ESMValCoreConfig(ValidatedConfig):
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


def _load_default_config(filename, drs_filename=None):
    mapping = read_config_file(filename)

    global config_default

    config_default.update(mapping)

    if drs_filename:
        drs_mapping = read_config_file(drs_filename)
        config_default.update(drs_mapping)


def _load_user_config(filename, raise_exception=True):
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


DEFAULT_CONFIG_DIR = Path(__file__).parent
DEFAULT_CONFIG = DEFAULT_CONFIG_DIR / 'config-default.yml'
DEFAULT_DRS = DEFAULT_CONFIG_DIR / 'drs-default.yml'

USER_CONFIG_DIR = Path.home() / '.esmvaltool'
USER_CONFIG = USER_CONFIG_DIR / 'config-user.yml'

# initialize placeholders
config_default = ESMValCoreConfig()
config = ESMValCoreConfig()
config_orig = ESMValCoreConfig()

# update config objects
_load_default_config(DEFAULT_CONFIG, DEFAULT_DRS)
_load_user_config(USER_CONFIG, raise_exception=False)
