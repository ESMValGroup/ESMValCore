"""Importable config object."""
from __future__ import annotations

import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import dask.config
import yaml

import esmvalcore
from esmvalcore.config._config_validators import (
    _deprecated_options_defaults,
    _deprecators,
    _validators,
)
from esmvalcore.config._validated_config import ValidatedConfig
from esmvalcore.exceptions import (
    ESMValCoreDeprecationWarning,
    InvalidConfigParameter,
)

URL = ('https://docs.esmvaltool.org/projects/'
       'ESMValCore/en/latest/quickstart/configure.html')


class Config(ValidatedConfig):
    """ESMValTool configuration object.

    Do not instantiate this class directly, but use
    :obj:`esmvalcore.config.CFG` instead.

    """
    # TODO: remove in v2.14.0
    _DEFAULT_USER_CONFIG_DIR = Path.home() / '.esmvaltool'

    _validate = _validators
    _deprecate = _deprecators
    _deprecated_defaults = _deprecated_options_defaults
    _warn_if_missing = (
        ('drs', URL),
        ('rootpath', URL),
    )

    def __init__(self, *args, **kwargs):
        """Initialize class instance."""
        super().__init__(*args, **kwargs)
        msg = (
            "Do not instantiate `Config` objects directly, this will lead "
            "to unexpected behavior. Use `esmvalcore.config.CFG` instead."
        )
        warnings.warn(msg, UserWarning)

    # TODO: remove in v2.14.0
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
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore',
                message="Do not instantiate `Config` objects directly",
                category=UserWarning,
                module='esmvalcore',
            )
            new = cls()
        new.update(Config._load_default_config())

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

    # TODO: remove in v2.14.0
    @classmethod
    def _load_default_config(cls):
        """Load the default configuration."""
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore',
                message="Do not instantiate `Config` objects directly",
                category=UserWarning,
                module='esmvalcore',
            )
            new = cls()

        paths = [
            Path(esmvalcore.__file__).parent / 'config' / 'config_defaults'
        ]
        mapping = dask.config.collect(paths=paths, env={})
        new.update(mapping)

        return new

    # TODO: remove in v2.14.0
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

    # TODO: remove in v2.14.0
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

    # TODO: remove in v2.14.0
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

    # TODO: remove in v2.14.0
    def load_from_file(
        self,
        filename: Optional[os.PathLike | str] = None,
    ) -> None:
        """Load user configuration from the given file."""
        msg = (
            "The method `CFG.load_from_file()` has been deprecated in "
            "ESMValCore version 2.12.0 and is scheduled for removal in "
            "version 2.14.0. Please update the `CFG` directly instead using "
            "`CFG.update()` or `CFG[...] = ...`."
        )
        warnings.warn(msg, ESMValCoreDeprecationWarning)
        self.clear()
        self.update(Config._load_user_config(filename))

    def reload(self):
        """Reload the original configuration object."""
        self.clear()
        paths = [str(p) for p in CONFIG_DIRS.values()]
        config_dict = dask.config.collect(paths=paths, env={})
        try:
            self.update(config_dict)
        except InvalidConfigParameter as exc:
            paths_str = '\n'.join(
                f'{v} ({k})' for (k, v) in CONFIG_DIRS.items()
            )
            raise InvalidConfigParameter(
                f"{str(exc)}\n\nThe following configuration directories have "
                f"been read:\n{paths_str}"
            )
        self.check_missing()

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
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore',
                message="Do not instantiate `Session` objects directly",
                category=UserWarning,
                module='esmvalcore',
            )
            session = Session(config=self.copy(), name=name)
        return session


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
        msg = (
            "Do not instantiate `Session` objects directly, this will lead "
            "to unexpected behavior. Use "
            "`esmvalcore.config.CFG.start_session` instead."
        )
        warnings.warn(msg, UserWarning)

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

    # TODO: remove in v2.14.0
    @property
    def config_dir(self):
        """Return user config directory."""
        msg = (
            "The attribute `Session.config_dir` has been deprecated in "
            "ESMValCore version 2.12.0 and is scheduled for removal in "
            "version 2.14.0."
        )
        warnings.warn(msg, ESMValCoreDeprecationWarning)
        if self.get('config_file') is None:
            return None
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


def _get_user_config_dir_from_cli() -> None | str:
    """Try to get user-defined configuration directory from CLI arguments.

    The hack of directly parsing the CLI arguments here (instead of using the
    `fire` or `argparser` module) ensures that the correct user configuration
    file is used. This will always work, regardless of when this module has
    been imported in the code.

    Note
    ----
    If not called within the `esmvaltool` program, always return `None`.

    """
    if Path(sys.argv[0]).name != 'esmvaltool':
        return None

    for arg in sys.argv:
        for opt in ('--config-dir', '--config_dir'):
            if opt in arg:
                # Parse '--config-dir=/dir' or '--config_dir=/dir'
                partition = arg.partition('=')
                if partition[2]:
                    return partition[2]

                # Parse '--config-dir /dir' or '--config_dir /dir'
                config_idx = sys.argv.index(opt)
                if config_idx == len(sys.argv) - 1:  # no dir given
                    return None
                return sys.argv[config_idx + 1]

    return None


def _get_user_config() -> tuple[str, Path]:
    """Get user configuration directory.

    The following directories are considered (sorted by priority):

    1. Internal `_ESMVALTOOL_USER_CONFIG_DIR_` environment variable (this
       ensures that any subprocess spawned by the esmvaltool program will use
       the correct user configuration directory).
    2. Command line arguments `--config-dir` or `--config_dir` (both variants
       are allowed by the fire module), but only if script name is
       `esmvaltool`.
    3. Default directory (`~/.config/esmvaltool`).

    Note
    ----
    If this function is used within the esmvaltool program, set the
    `_ESMVALTOOL_USER_CONFIG_DIR_` at the end of this method to make sure that
    subsequent calls of this method (also in suprocesses) use the correct user
    configuration directory.

    """
    # (1) Internal _ESMVALTOOL_USER_CONFIG_FILE_ environment variable
    source = '_ESMVALTOOL_USER_CONFIG_DIR_ environment variable'
    config_dir: None | str | Path = os.getenv('_ESMVALTOOL_USER_CONFIG_DIR_')

    # (2) CLI arguments
    if config_dir is None:
        source = 'command line argument'
        config_dir = _get_user_config_dir_from_cli()

    # (3) Default location
    if config_dir is None:
        source = 'default user configuration directory'
        config_dir = Path.home() / '.config' / 'esmvaltool'

    config_dir = Path(config_dir).expanduser().absolute()

    # If used within the esmvaltool program, make sure that subsequent calls of
    # this method (also in suprocesses) use the correct user configuration dir
    if Path(sys.argv[0]).name == 'esmvaltool':
        os.environ['_ESMVALTOOL_USER_CONFIG_DIR_'] = str(config_dir)

    return (source, config_dir)


def _get_config_dirs() -> dict[str, Path]:
    """Get all configuration directories."""
    # Defaults (lowest priority)
    config_dirs: dict[str, Path] = {
        'defaults': Path(__file__).parent / 'config_defaults',
    }

    # User input (medium priority)
    config_user = _get_user_config()
    config_dirs[config_user[0]] = config_user[1]

    # Environoment variable (highest priority)
    if 'ESMVALTOOL_CONFIG_DIR' in os.environ:
        config_dirs['ESMVALTOOL_CONFIG_DIR environment variable'] = (
            Path(os.environ['ESMVALTOOL_CONFIG_DIR']).expanduser().absolute()
        )

    # Check existence, except for default user configuration dir
    for (source, config_dir) in config_dirs.items():
        if source == 'default user configuration directory':
            continue
        if not config_dir.is_dir():
            raise NotADirectoryError(
                f"Configuration directory {config_dir} specified via {source} "
                f"is not a valid directory"
            )

    return config_dirs


def get_global_config():
    """Get configuration object from global paths."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message="Do not instantiate `Config` objects directly",
            category=UserWarning,
            module='esmvalcore',
        )
        config_obj = Config()
    config_obj.reload()
    return config_obj


# Deprecated way of specifying configuration (remove in v2.14.0)
_DEPRECATIONS = []
_deprecated_config_user_path = Config._get_config_user_path()
if _deprecated_config_user_path.is_file():
    deprecation_msg = (
        f"Usage of the single configuration file "
        f"~/.esmvaltool/config-user.yml or specifying it via CLI argument "
        f"`--config_file` has been deprecated in ESMValCore version 2.12.0 "
        f"and is scheduled for removal in version 2.14.0. Please run "
        f"`mkdir -p ~/.config/esmvaltool && mv {_deprecated_config_user_path} "
        f"~/.config/esmvaltool` (or alternatively use a custom "
        f"`--config_dir`) and omit `--config_file`."
    )
    warnings.warn(deprecation_msg, ESMValCoreDeprecationWarning)
    _DEPRECATIONS.append(deprecation_msg)
    CONFIG_DIRS = {
        'defaults': Path(__file__).parent / 'config_defaults',
        'single configuration file [deprecated]': _deprecated_config_user_path,
    }
    CFG = Config._load_user_config(raise_exception=False)

# New way of specifying configuration
# Initialize configuration objects
else:
    CONFIG_DIRS = _get_config_dirs()
    CFG = get_global_config()
