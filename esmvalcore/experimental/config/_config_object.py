"""Importable config object."""

from datetime import datetime
from pathlib import Path

import yaml

import esmvalcore

from ._config_validators import _validators
from ._validated_config import ValidatedConfig


class Config(ValidatedConfig):
    """ESMValTool configuration object."""

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
                raise FileNotFoundError(f'Cannot find: `{filename}`'
                                        f'locally or in `{try_path}`')

        _load_user_config(path)

    def reload(self):
        """Reload the config file."""
        filename = self.get('config_file', DEFAULT_CONFIG)
        self.load_from_file(filename)

    def start_session(self, name: str):
        """Start a new session from this configuration object.

        Parameters
        ----------
        name: str
            Name of the session.

        Returns
        -------
        Session
        """
        return Session(config=self.copy(), name=name)


class Session(ValidatedConfig):
    """Container class for session configuration and directory information.

    Parameters
    ----------
    config : dict
        Dictionary with configuration settings.
    name : str
        Name of the session to initialize, for example, the name of the
        recipe (default='session').
    """

    validate = _validators

    def __init__(self, config: dict, name: str = 'session'):
        super().__init__(config)
        self.init_session_dir(name)

    def init_session_dir(self, name: str = 'session'):
        """Initialize session.

        The `name` is used to name the working directory, e.g.
        `recipe_example_20200916/`. If no name is given, such as in an
        interactive session, defaults to `session`.
        """
        now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        session_name = f"{name}_{now}"
        self._session_dir = self['output_dir'] / session_name

    @property
    def session_dir(self):
        """Return session directory."""
        return self._session_dir

    @property
    def preproc_dir(self):
        """Return preproc directory."""
        return self.session_dir / 'preproc'

    @property
    def work_dir(self):
        """Return work directory."""
        return self.session_dir / 'work'

    @property
    def plot_dir(self):
        """Return plot directory."""
        return self.session_dir / 'plots'

    @property
    def run_dir(self):
        """Return run directory."""
        return self.session_dir / 'run'

    @property
    def config_dir(self):
        """Return user config directory."""
        return USER_CONFIG_DIR


def read_config_file(config_file):
    """Read config user file and store settings in a dictionary."""
    config_file = Path(config_file)
    if not config_file.exists():
        raise IOError(f'Config file `{config_file}` does not exist.')

    with open(config_file, 'r') as file:
        cfg = yaml.safe_load(file)

    return cfg


def _load_default_config(filename: str):
    """Load the default configuration."""
    mapping = read_config_file(filename)

    global CFG_DEFAULT

    CFG_DEFAULT.update(mapping)


def _load_user_config(filename: str, raise_exception: bool = True):
    """Load user configuration from the given file.

    The config is cleared and updated in-place.

    Parameters
    ----------
    filename: pathlike
        Name of the config file, must be yaml format
    raise_exception : bool
        Raise an exception if `filename` can not be found (default).
        Otherwise, silently pass and use the default configuration. This
        setting is necessary for the case where `.esmvaltool/config-user.yml`
        has not been defined (i.e. first start).
    """
    try:
        mapping = read_config_file(filename)
        mapping['config_file'] = filename
    except IOError:
        if raise_exception:
            raise
        mapping = {}

    global CFG

    CFG.clear()
    CFG.update(CFG_DEFAULT)
    CFG.update(mapping)


DEFAULT_CONFIG_DIR = Path(esmvalcore.__file__).parent
DEFAULT_CONFIG = DEFAULT_CONFIG_DIR / 'config-user.yml'

USER_CONFIG_DIR = Path.home() / '.esmvaltool'
USER_CONFIG = USER_CONFIG_DIR / 'config-user.yml'

# initialize placeholders
CFG_DEFAULT = Config()
CFG = Config()

# update config objects
_load_default_config(DEFAULT_CONFIG)
_load_user_config(USER_CONFIG, raise_exception=False)
