"""Importable config object."""

from datetime import datetime
from pathlib import Path

import yaml

import esmvalcore

from ._config_validators import _validators
from ._validated_config import ValidatedConfig

URL = ('https://docs.esmvaltool.org/projects/'
       'ESMValCore/en/latest/quickstart/configure.html')


class Config(ValidatedConfig):
    """ESMValTool configuration object.

    Do not instantiate this class directly, but use
    :obj:`esmvalcore.experimental.CFG` instead.
    """

    _validate = _validators
    _warn_if_missing = (
        ('drs', URL),
        ('rootpath', URL),
    )

    @classmethod
    def _load_user_config(cls, filename: str, raise_exception: bool = True):
        """Load user configuration from the given file.

        The config is cleared and updated in-place.

        Parameters
        ----------
        filename: pathlike
            Name of the config file, must be yaml format
        raise_exception : bool
            Raise an exception if `filename` can not be found (default).
            Otherwise, silently pass and use the default configuration. This
            setting is necessary for the case where
            `.esmvaltool/config-user.yml` has not been defined (i.e. first
            start).
        """
        new = cls()

        try:
            mapping = _read_config_file(filename)
            mapping['config_file'] = filename
        except IOError:
            if raise_exception:
                raise
            mapping = {}

        new.update(CFG_DEFAULT)
        new.update(mapping)
        new.check_missing()

        return new

    @classmethod
    def _load_default_config(cls, filename: str):
        """Load the default configuration."""
        new = cls()

        mapping = _read_config_file(filename)
        new.update(mapping)

        return new

    def load_from_file(self, filename):
        """Load user configuration from the given file."""
        path = Path(filename).expanduser()
        if not path.exists():
            try_path = USER_CONFIG_DIR / filename
            if try_path.exists():
                path = try_path
            else:
                raise FileNotFoundError(f'Cannot find: `{filename}`'
                                        f'locally or in `{try_path}`')

        self.clear()
        self.update(Config._load_user_config(path))

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

    Do not instantiate this class directly, but use
    :obj:`CFG.start_session` instead.

    Parameters
    ----------
    config : dict
        Dictionary with configuration settings.
    name : str
        Name of the session to initialize, for example, the name of the
        recipe (default='session').
    """

    _validate = _validators

    def __init__(self, config: dict, name: str = 'session'):
        super().__init__(config)
        self.set_session_name(name)

    def set_session_name(self, name: str = 'session'):
        """Set the name for the session.

        The `name` is used to name the session directory, e.g.
        `session_20201208_132800/`. The date is suffixed automatically.
        """
        now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.session_name = f"{name}_{now}"

    @property
    def session_dir(self):
        """Return session directory."""
        return self['output_dir'] / self.session_name

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

    def to_config_user(self) -> dict:
        """Turn the `Session` object into a recipe-compatible dict.

        This dict is compatible with the `config-user` argument in
        :obj:`esmvalcore._recipe.Recipe`.
        """
        dct = self.copy()
        dct['run_dir'] = self.run_dir
        dct['work_dir'] = self.work_dir
        dct['preproc_dir'] = self.preproc_dir
        dct['plot_dir'] = self.plot_dir
        return dct


def _read_config_file(config_file):
    """Read config user file and store settings in a dictionary."""
    config_file = Path(config_file)
    if not config_file.exists():
        raise IOError(f'Config file `{config_file}` does not exist.')

    with open(config_file, 'r') as file:
        cfg = yaml.safe_load(file)

    return cfg


DEFAULT_CONFIG_DIR = Path(esmvalcore.__file__).parent
DEFAULT_CONFIG = DEFAULT_CONFIG_DIR / 'config-user.yml'

USER_CONFIG_DIR = Path.home() / '.esmvaltool'
USER_CONFIG = USER_CONFIG_DIR / 'config-user.yml'

# initialize placeholders
CFG_DEFAULT = Config._load_default_config(DEFAULT_CONFIG)
CFG = Config._load_user_config(USER_CONFIG, raise_exception=False)
