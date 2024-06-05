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
from esmvalcore.exceptions import InvalidConfigParameter

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

        config_user_path = cls._get_config_user_path(filename)

        try:
            mapping = cls._read_config_file(config_user_path)
            mapping['config_file'] = config_user_path
        except FileNotFoundError:
            if raise_exception:
                raise
            mapping = {}

        try:
            new.update(mapping)
            new.check_missing()
        except InvalidConfigParameter as exc:
            raise InvalidConfigParameter(
                f"Failed to parse user configuration file {config_user_path}: "
                f"{str(exc)}"
            ) from exc

        return new

    @classmethod
    def _load_default_config(cls):
        """Load the default configuration."""
        new = cls()

        package_config_user_path = Path(
            esmvalcore.__file__
        ).parent / 'config-user.yml'
        mapping = cls._read_config_file(package_config_user_path)

        # Add defaults that are not available in esmvalcore/config-user.yml
        mapping['check_level'] = CheckLevels.DEFAULT
        mapping['config_file'] = package_config_user_path
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
        """Read configuration file and store settings in a dictionary."""
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

        `filename` can be given as absolute or relative path. In the latter
        case, search in the current working directory and `~/.esmvaltool` (in
        that order).

        If `filename` is not given, try to get user configuration file from the
        following locations (sorted by descending priority):

        1. Internal `_ESMVALTOOL_USER_CONFIG_FILE_` environment variable
           (this ensures that any subprocess spawned by the esmvaltool program
           will use the correct user configuration file).
        2. Command line arguments `--config-file` or `--config_file` (both
           variants are allowed by the fire module), but only if script name is
           `esmvaltool`.
        3. `config-user.yml` within default ESMValTool configuration directory
           `~/.esmvaltool`.

        Note
        ----
        This will NOT check if the returned file actually exists to allow
        loading the module without any configuration file (this is relevant if
        the module is used within a script or notebook). To check if the file
        actually exists, use the method `load_from_file` (this is done when
        using the `esmvaltool` CLI).

        If used within the esmvaltool program, set the
        _ESMVALTOOL_USER_CONFIG_FILE_ at the end of this method to make sure
        that subsequent calls of this method (also in suprocesses) use the
        correct user configuration file.

        """
        # (1) Try to get user configuration file from `filename` argument
        config_user = filename

        # (2) Try to get user configuration file from internal
        # _ESMVALTOOL_USER_CONFIG_FILE_ environment variable
        if (
                config_user is None and
                '_ESMVALTOOL_USER_CONFIG_FILE_' in os.environ
        ):
            config_user = os.environ['_ESMVALTOOL_USER_CONFIG_FILE_']

        # (3) Try to get user configuration file from CLI arguments
        if config_user is None:
            config_user = Config._get_config_path_from_cli()

        # (4) Default location
        if config_user is None:
            config_user = Config._DEFAULT_USER_CONFIG_DIR / 'config-user.yml'

        config_user = Path(config_user).expanduser()

        # Also search path relative to ~/.esmvaltool if necessary
        if not (config_user.is_file() or config_user.is_absolute()):
            config_user = Config._DEFAULT_USER_CONFIG_DIR / config_user
        config_user = config_user.absolute()

        # If used within the esmvaltool program, make sure that subsequent
        # calls of this method (also in suprocesses) use the correct user
        # configuration file
        if Path(sys.argv[0]).name == 'esmvaltool':
            os.environ['_ESMVALTOOL_USER_CONFIG_FILE_'] = str(config_user)

        return config_user

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
                    if config_idx == len(sys.argv) - 1:  # no file given
                        return None
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
                "missing; make sure to only use the `CFG` object from the "
                "`esmvalcore.config` module"
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
    relative_cmor_log = Path('run', 'cmor_log.txt')
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
    def cmor_log(self):
        """Return CMOR log file."""
        return self.session_dir / self.relative_cmor_log

    @property
    def _fixed_file_dir(self):
        """Return fixed file directory."""
        return self.session_dir / self._relative_fixed_file_dir


# Initialize configuration objects
CFG_DEFAULT = MappingProxyType(Config._load_default_config())
CFG = Config._load_user_config(raise_exception=False)
