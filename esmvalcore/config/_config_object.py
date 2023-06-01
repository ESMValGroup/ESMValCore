"""Importable config object."""

import os
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Optional, Union

import yaml

import esmvalcore
from esmvalcore.cmor.check import CheckLevels

from ._config_validators import (
    _deprecated_options_defaults,
    _deprecators,
    _validators,
)
from ._validated_config import ValidatedConfig

URL = ('https://docs.esmvaltool.org/projects/'
       'ESMValCore/en/latest/quickstart/configure.html')


class Config(ValidatedConfig):
    """ESMValTool configuration object.

    Do not instantiate this class directly, but use
    :obj:`esmvalcore.config.CFG` instead.
    """

    _validate = _validators
    _deprecate = _deprecators
    _deprecated_defaults = _deprecated_options_defaults
    _warn_if_missing = (
        ('drs', URL),
        ('rootpath', URL),
    )

    @classmethod
    def _load_user_config(cls,
                          filename: Union[os.PathLike, str],
                          raise_exception: bool = True):
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
    def _load_default_config(cls, filename: Union[os.PathLike, str]):
        """Load the default configuration."""
        new = cls()

        mapping = _read_config_file(filename)
        # Add defaults that are not available in esmvalcore/config-user.yml
        mapping['check_level'] = CheckLevels.DEFAULT
        mapping['config_file'] = filename
        mapping['diagnostics'] = None
        mapping['extra_facets_dir'] = tuple()
        mapping['max_datasets'] = None
        mapping['max_years'] = None
        mapping['resume_from'] = []
        mapping['run_diagnostic'] = True
        mapping['skip_nonexistent'] = False

        new.update(mapping)

        return new

    def load_from_file(
        self,
        filename: Optional[Union[os.PathLike, str]] = None,
    ) -> None:
        """Load user configuration from the given file."""
        if filename is None:
            filename = USER_CONFIG
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
    _deprecate = _deprecators
    _deprecated_defaults = _deprecated_options_defaults

    relative_preproc_dir = Path('preproc')
    relative_work_dir = Path('work')
    relative_plot_dir = Path('plots')
    relative_run_dir = Path('run')
    relative_main_log = Path('run', 'main_log.txt')
    relative_main_log_debug = Path('run', 'main_log_debug.txt')
    _relative_fixed_file_dir = Path('preproc', 'fixed_files')

    def __init__(self, config: dict, name: str = 'session'):
        super().__init__(config)
        self.session_name: Union[str, None] = None
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
        return self.session_dir / self.relative_preproc_dir

    @property
    def work_dir(self):
        """Return work directory."""
        return self.session_dir / self.relative_work_dir

    @property
    def plot_dir(self):
        """Return plot directory."""
        return self.session_dir / self.relative_plot_dir

    @property
    def run_dir(self):
        """Return run directory."""
        return self.session_dir / self.relative_run_dir

    @property
    def config_dir(self):
        """Return user config directory."""
        return USER_CONFIG_DIR

    @property
    def main_log(self):
        """Return main log file."""
        return self.session_dir / self.relative_main_log

    @property
    def main_log_debug(self):
        """Return main log debug file."""
        return self.session_dir / self.relative_main_log_debug

    @property
    def _fixed_file_dir(self):
        """Return fixed file directory."""
        return self.session_dir / self._relative_fixed_file_dir


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
CFG_DEFAULT = MappingProxyType(Config._load_default_config(DEFAULT_CONFIG))
CFG = Config._load_user_config(USER_CONFIG, raise_exception=False)
