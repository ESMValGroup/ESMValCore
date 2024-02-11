"""Importable config object."""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Optional

import yaml

import esmvalcore
from esmvalcore.cmor.check import CheckLevels
from esmvalcore.exceptions import ConfigParserError, InvalidConfigParameter

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
    _DEFAULT_USER_CONFIG_DIR = Path.home() / '.esmvaltool'

    _validate = _validators
    _deprecate = _deprecators
    _deprecated_defaults = _deprecated_options_defaults
    _warn_if_missing = (
        ('drs', URL),
        ('rootpath', URL),
    )

    @classmethod
    def _load_user_config(
        cls,
        filename: Optional[os.PathLike | str] = None,
        raise_exception: bool = True,
    ):
        """Load user configuration from the given file.

        The config is cleared and updated in-place.

        Parameters
        ----------
        filename:
            Name of the user configuration file (must be YAML format). If
            `None`, use the rules given in `Config._get_config_user_path` to
            determine the path.
        raise_exception : bool
            If ``True``, raise an exception if `filename` cannot be found.  If
            ``False``, silently pass and use the default configuration. This
            setting is necessary during the loading of this module when no
            configuration file is given (relevant if used within a script or
            notebook).
        """
        new = cls()
        new.update(CFG_DEFAULT)

        config_user_path = Config._get_config_user_path(filename)

        try:
            mapping = Config._read_config_file(config_user_path)
            mapping['config_file'] = config_user_path
        except FileNotFoundError:
            if raise_exception:
                raise
            mapping = {}

        try:
            new.update(mapping)
            new.check_missing()
        except InvalidConfigParameter as exc:
            raise ConfigParserError(
                f"Failed to parse user configuration file {config_user_path}: "
                f"{str(exc)}"
            ) from exc

        return new

    @classmethod
    def _load_default_config(cls):
        """Load the default configuration."""
        new = cls()

        config_user_path = Path(esmvalcore.__file__).parent / 'config-user.yml'
        mapping = cls._read_config_file(config_user_path)

        # Add defaults that are not available in esmvalcore/config-user.yml
        mapping['check_level'] = CheckLevels.DEFAULT
        mapping['config_file'] = config_user_path
        mapping['diagnostics'] = None
        mapping['extra_facets_dir'] = tuple()
        mapping['max_datasets'] = None
        mapping['max_years'] = None
        mapping['resume_from'] = []
        mapping['run_diagnostic'] = True
        mapping['skip_nonexistent'] = False

        new.update(mapping)

        return new

    @staticmethod
    def _read_config_file(config_user_path: Path) -> dict:
        """Read config user file and store settings in a dictionary."""
        if not config_user_path.is_file():
            raise FileNotFoundError(
                f"Config file '{config_user_path}' does not exist"
            )

        with open(config_user_path, 'r', encoding='utf-8') as file:
            cfg = yaml.safe_load(file)

        return cfg

    @staticmethod
    def _get_config_user_path(
        filename: Optional[os.PathLike | str] = None
    ) -> Path:
        """Get path to user configuration file.

        `filename` can be given as absolute path, or as path relative to the
        directory specified with `ESMVALTOOL_CONFIG` (if present) or
        `~/.esmvaltool`.

        If `filename` is not given, try to get user configuration file from the
        following locations (sorted by descending priority):

        1. Command line arguments `--config-file` or `--config_file`.
        2. `config-user.yml` within ESMValTool configuration directory given by
        `ESMVALTOOL_CONFIG` environment variable.
        3. `config-user.yml` within default ESMValTool configuration directory
        `~/.esmvaltool`.

        Note
        ----
        This will NOT check if the returned file actually exists to allow
        loading the module without any configuration file (this is relevant if
        the module is used within a script or notebook). To check if the file
        actually exists, use the method `load_from_file` (this is done when
        using the `esmvaltool` CLI).

        """
        config_user: None | os.PathLike | str = None

        # (1) Try to get config user file from `filename` argument (also test
        # relative to ESMVALTOOL_CONFIG or ~/.esmvaltool)
        if filename is not None:
            config_user = Config._get_config_path_from_arg(filename)

        # (2) Try to get config user file from CLI arguments (if file
        # specified, raise error if it does not exist)
        if config_user is None:
            config_user = Config._get_config_path_from_cli()

        # (3) Environment variable (if specified but file does not exist, do
        # not raise an error)
        if config_user is None and 'ESMVALTOOL_CONFIG' in os.environ:
            config_user = Path(
                os.environ['ESMVALTOOL_CONFIG']
            ) / 'config-user.yml'

        # (4) Default location (do not raise an error if file does not exist)
        if config_user is None:
            config_user = Config._DEFAULT_USER_CONFIG_DIR / 'config-user.yml'

        return Path(config_user).expanduser().absolute()

    @staticmethod
    def _get_config_path_from_arg(filename: os.PathLike | str) -> None | Path:
        """Try to get configuration path from argument.

        `filename` can be given as absolute path, or as path relative to the
        directory specified with `ESMVALTOOL_CONFIG` (if present) or
        `~/.esmvaltool`.

        Note
        ----
        Does not check if file exists.

        """
        filename = Path(filename).expanduser()
        if filename.is_file():
            return filename
        if 'ESMVALTOOL_CONFIG' in os.environ:
            return Path(os.environ['ESMVALTOOL_CONFIG']) / filename
        return Config._DEFAULT_USER_CONFIG_DIR / filename

    @staticmethod
    def _get_config_path_from_cli() -> None | str:
        """Try to get configuration path from CLI arguments.

        The hack of directly parsing the CLI arguments here (instead of using
        the fire or argparser module) ensures that the correct user
        configuration file is used. This will always work, regardless of when
        this module has been imported in the code.

        Note
        ----
        This only works if the script name is `esmvaltool`. Does not check if
        file exists.

        """
        if Path(sys.argv[0]).name != 'esmvaltool':
            return None

        for arg in sys.argv:
            for opt in ('--config-file', '--config_file'):
                if opt in arg:
                    # Parse '--config-file=/file.yml' or
                    # '--config_file=/file.yml'
                    partition = arg.partition('=')
                    if partition[2]:
                        return partition[2]

                    # Parse '--config-file /file.yml' or
                    # '--config_file /file.yml'
                    config_idx = sys.argv.index(opt)
                    return sys.argv[config_idx + 1]

        return None

    def load_from_file(
        self,
        filename: Optional[os.PathLike | str] = None,
    ) -> None:
        """Load user configuration from the given file."""
        self.clear()
        self.update(Config._load_user_config(filename))

    def reload(self):
        """Reload the config file."""
        if 'config_file' not in self:
            raise ValueError(
                "Cannot reload configuration, option 'config_file' is "
                "missing; make sure to only initialize this object with "
                "`Config._load_user_config()` or "
                "`Config._load_default_config()`"
            )
        self.load_from_file(self['config_file'])

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
        self.session_name: str | None = None
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
        return Path(self['config_file']).parent

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


# Initialize configuration objects
CFG_DEFAULT = MappingProxyType(Config._load_default_config())
CFG = Config._load_user_config(raise_exception=False)
