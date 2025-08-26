"""Importable config object."""

from __future__ import annotations

import datetime
import os
import sys
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

URL = (
    "https://docs.esmvaltool.org/projects/"
    "ESMValCore/en/latest/quickstart/configure.html"
)

# Configuration directory in which defaults are stored
DEFAULT_CONFIG_DIR = (
    Path(esmvalcore.__file__).parent / "config" / "configurations" / "defaults"
)


def _get_user_config_dir() -> Path:
    """Get user configuration directory."""
    if "ESMVALTOOL_CONFIG_DIR" in os.environ:
        user_config_dir = (
            Path(os.environ["ESMVALTOOL_CONFIG_DIR"]).expanduser().absolute()
        )
        if not user_config_dir.is_dir():
            msg = (
                f"Invalid configuration directory specified via "
                f"ESMVALTOOL_CONFIG_DIR environment variable: "
                f"{user_config_dir} is not an existing directory"
            )
            raise NotADirectoryError(
                msg,
            )
        return user_config_dir
    return Path.home() / ".config" / "esmvaltool"


def _get_user_config_source() -> str:
    """Get source of user configuration directory."""
    if "ESMVALTOOL_CONFIG_DIR" in os.environ:
        return "ESMVALTOOL_CONFIG_DIR environment variable"
    return "default user configuration directory"


# User configuration directory
USER_CONFIG_DIR = _get_user_config_dir()

# Source of user configuration directory
USER_CONFIG_SOURCE = _get_user_config_source()


class Config(ValidatedConfig):
    """ESMValTool configuration object.

    Do not instantiate this class directly, but use
    :obj:`esmvalcore.config.CFG` instead.

    """

    # TODO: remove in v2.14.0
    _DEFAULT_USER_CONFIG_DIR = Path.home() / ".esmvaltool"

    _validate = _validators
    _deprecate = _deprecators
    _deprecated_defaults = _deprecated_options_defaults
    _warn_if_missing = (
        ("drs", URL),
        ("rootpath", URL),
    )

    def __init__(self, *args, **kwargs):
        """Initialize class instance."""
        super().__init__(*args, **kwargs)
        msg = (
            "Do not instantiate `Config` objects directly, this will lead "
            "to unexpected behavior. Use `esmvalcore.config.CFG` instead."
        )
        warnings.warn(msg, UserWarning, stacklevel=2)

    # TODO: remove in v2.14.0
    @classmethod
    def _load_user_config(
        cls,
        filename: os.PathLike | str | None = None,
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
                "ignore",
                message="Do not instantiate `Config` objects directly",
                category=UserWarning,
                module="esmvalcore",
            )
            new = cls()
        new.update(Config._load_default_config())

        config_user_path = cls._get_config_user_path(filename)

        try:
            mapping = cls._read_config_file(config_user_path)
            mapping["config_file"] = config_user_path
        except FileNotFoundError:
            if raise_exception:
                raise
            mapping = {}

        try:
            new.update(mapping)
            new.check_missing()
        except InvalidConfigParameter as exc:
            msg = (
                f"Failed to parse user configuration file {config_user_path}: "
                f"{exc!s}"
            )
            raise InvalidConfigParameter(msg) from exc

        return new

    # TODO: remove in v2.14.0
    @classmethod
    def _load_default_config(cls):
        """Load the default configuration."""
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Do not instantiate `Config` objects directly",
                category=UserWarning,
                module="esmvalcore",
            )
            new = cls()

        paths = [DEFAULT_CONFIG_DIR]
        mapping = dask.config.collect(paths=paths, env={})
        new.update(mapping)

        return new

    # TODO: remove in v2.14.0
    @staticmethod
    def _read_config_file(config_user_path: Path) -> dict:
        """Read configuration file and store settings in a dictionary."""
        if not config_user_path.is_file():
            msg = f"Config file '{config_user_path}' does not exist"
            raise FileNotFoundError(
                msg,
            )

        with open(config_user_path, encoding="utf-8") as file:
            return yaml.safe_load(file)

    # TODO: remove in v2.14.0
    @staticmethod
    def _get_config_user_path(
        filename: os.PathLike | str | None = None,
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
            config_user is None
            and "_ESMVALTOOL_USER_CONFIG_FILE_" in os.environ
        ):
            config_user = os.environ["_ESMVALTOOL_USER_CONFIG_FILE_"]

        # (3) Try to get user configuration file from CLI arguments
        if config_user is None:
            config_user = Config._get_config_path_from_cli()

        # (4) Default location
        if config_user is None:
            config_user = Config._DEFAULT_USER_CONFIG_DIR / "config-user.yml"

        config_user = Path(config_user).expanduser()

        # Also search path relative to ~/.esmvaltool if necessary
        if not (config_user.is_file() or config_user.is_absolute()):
            config_user = Config._DEFAULT_USER_CONFIG_DIR / config_user
        config_user = config_user.absolute()

        # If used within the esmvaltool program, make sure that subsequent
        # calls of this method (also in suprocesses) use the correct user
        # configuration file
        if Path(sys.argv[0]).name == "esmvaltool":
            os.environ["_ESMVALTOOL_USER_CONFIG_FILE_"] = str(config_user)

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
        if Path(sys.argv[0]).name != "esmvaltool":
            return None

        for arg in sys.argv:
            for opt in ("--config-file", "--config_file"):
                if opt in arg:
                    # Parse '--config-file=/file.yml' or '--config_file=/file.yml'
                    partition = arg.partition("=")
                    if partition[2]:
                        return partition[2]

                    # Parse '--config-file /file.yml' or '--config_file /file.yml'
                    config_idx = sys.argv.index(opt)
                    if config_idx == len(sys.argv) - 1:  # no file given
                        return None
                    return sys.argv[config_idx + 1]

        return None

    # TODO: remove in v2.14.0
    def load_from_file(
        self,
        filename: os.PathLike | str | None = None,
    ) -> None:
        """Load user configuration from the given file.

        .. deprecated:: 2.12.0
            This method has been deprecated in ESMValCore version 2.14.0 and is
            scheduled for removal in version 2.14.0. Please use
            `CFG.load_from_dirs()` instead.

        Parameters
        ----------
        filename:
            YAML file to load.

        """
        msg = (
            "The method `CFG.load_from_file()` has been deprecated in "
            "ESMValCore version 2.12.0 and is scheduled for removal in "
            "version 2.14.0. Please use `CFG.load_from_dirs()` instead."
        )
        warnings.warn(msg, ESMValCoreDeprecationWarning, stacklevel=2)
        self.clear()
        self.update(Config._load_user_config(filename))

    @staticmethod
    def _get_config_dict_from_dirs(dirs: Iterable[str | Path]) -> dict:
        """Get configuration :obj:`dict` from directories."""
        dirs_str: list[str] = []
        for config_dir in dirs:
            abs_config_dir = Path(config_dir).expanduser().absolute()
            dirs_str.append(str(abs_config_dir))
        return dask.config.collect(paths=dirs_str, env={})

    def load_from_dirs(self, dirs: Iterable[str | Path]) -> None:
        """Clear and load configuration object from directories.

        This searches for all YAML files within the given directories and
        merges them together using :func:`dask.config.collect`. Nested objects
        are properly considered; see :func:`dask.config.update` for details.
        Values in the latter directories are preferred to those in the former.

        Options that are not explicitly specified via YAML files are set to the
        :ref:`default values <config_options>`.

        Note
        ----
        Just like :func:`dask.config.collect`, this silently ignores
        non-existing directories.

        Parameters
        ----------
        dirs:
            A list of directories to search for YAML configuration files.

        Raises
        ------
        esmvalcore.exceptions.InvalidConfigParameter
            Invalid configuration option given.

        """
        # Always consider default options; these have the lowest priority
        dirs = [DEFAULT_CONFIG_DIR, *list(dirs)]

        new_config_dict = self._get_config_dict_from_dirs(dirs)
        self.clear()
        self.update(new_config_dict)

        self.check_missing()

    def reload(self) -> None:
        """Clear and reload the configuration object.

        This will read all YAML files in the user configuration directory (by
        default ``~/.config/esmvaltool``, but this can be changed with the
        ``ESMVALTOOL_CONFIG_DIR`` environment variable) and merges them
        together using :func:`dask.config.collect`. Nested objects are properly
        considered; see :func:`dask.config.update` for details.

        Options that are not explicitly specified via YAML files are set to the
        :ref:`default values <config_options>`.

        Note
        ----
        If the user configuration directory does not exist, this will be
        silently ignored.

        Raises
        ------
        esmvalcore.exceptions.InvalidConfigParameter
            Invalid configuration option given.

        """
        # TODO: remove in v2.14.0
        self.clear()
        _deprecated_config_user_path = Config._get_config_user_path()
        if _deprecated_config_user_path.is_file() and not os.environ.get(
            "ESMVALTOOL_CONFIG_DIR",
        ):
            deprecation_msg = (
                f"Usage of the single configuration file "
                f"~/.esmvaltool/config-user.yml or specifying it via CLI "
                f"argument `--config_file` has been deprecated in ESMValCore "
                f"version 2.12.0 and is scheduled for removal in version "
                f"2.14.0. To switch to the new configuration system, (1) run "
                f"`mkdir -p ~/.config/esmvaltool && mv "
                f"{_deprecated_config_user_path} ~/.config/esmvaltool` (or "
                f"alternatively use a custom `--config_dir`) and omit "
                f"`--config_file`, or (2) use the environment variable "
                f"ESMVALTOOL_CONFIG_DIR to specify a custom user "
                f"configuration directory. New configuration files present at "
                f"~/.config/esmvaltool or specified via `--config_dir` are "
                f"currently ignored."
            )
            warnings.warn(
                deprecation_msg,
                ESMValCoreDeprecationWarning,
                stacklevel=2,
            )
            self.update(Config._load_user_config(raise_exception=False))
            return

        # New since v2.12.0
        try:
            self.load_from_dirs([USER_CONFIG_DIR])
        except InvalidConfigParameter as exc:
            msg = (
                f"Failed to parse configuration directory {USER_CONFIG_DIR} "
                f"({USER_CONFIG_SOURCE}): {exc!s}"
            )
            raise InvalidConfigParameter(msg) from exc

    def start_session(self, name: str) -> Session:
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
                "ignore",
                message="Do not instantiate `Session` objects directly",
                category=UserWarning,
                module="esmvalcore",
            )
            return Session(config=self.copy(), name=name)

    def update_from_dirs(self, dirs: Iterable[str | Path]) -> None:
        """Update configuration object from directories.

        This will first search for all YAML files within the given directories
        and merge them together using :func:`dask.config.collect` (if identical
        values are provided in multiple files, the value from the last file
        will be used).  Then, the current configuration is merged with these
        new configuration options using :func:`dask.config.merge` (new values
        are preferred over old values). Nested objects are properly considered;
        see :func:`dask.config.update` for details.

        Note
        ----
        Just like :func:`dask.config.collect`, this silently ignores
        non-existing directories.

        Parameters
        ----------
        dirs:
            A list of directories to search for YAML configuration files.

        Raises
        ------
        esmvalcore.exceptions.InvalidConfigParameter
            Invalid configuration option given.

        """
        new_config_dict = self._get_config_dict_from_dirs(dirs)
        self.nested_update(new_config_dict)

    def nested_update(self, new_options: Mapping) -> None:
        """Nested update of configuration object with another mapping.

        Merge the existing configuration object with a new mapping using
        :func:`dask.config.merge`  (new values are preferred over old values).
        Nested objects are properly considered; see :func:`dask.config.update`
        for details.

        Parameters
        ----------
        new_options:
            New configuration options.

        Raises
        ------
        esmvalcore.exceptions.InvalidConfigParameter
            Invalid configuration option given.

        """
        merged_config_dict = dask.config.merge(self, new_options)
        self.update(merged_config_dict)
        self.check_missing()

    clear = ValidatedConfig.clear  # to show this is in API doc

    context = ValidatedConfig.context  # to show this is in API doc

    copy = ValidatedConfig.copy  # to show this is in API doc


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

    relative_preproc_dir = Path("preproc")
    relative_work_dir = Path("work")
    relative_plot_dir = Path("plots")
    relative_run_dir = Path("run")
    relative_main_log = Path("run", "main_log.txt")
    relative_main_log_debug = Path("run", "main_log_debug.txt")
    relative_cmor_log = Path("run", "cmor_log.txt")
    _relative_fixed_file_dir = Path("preproc", "fixed_files")

    def __init__(self, config: dict, name: str = "session"):
        super().__init__(config)
        self.session_name: str | None = None
        self.set_session_name(name)
        msg = (
            "Do not instantiate `Session` objects directly, this will lead "
            "to unexpected behavior. Use "
            "`esmvalcore.config.CFG.start_session` instead."
        )
        warnings.warn(msg, UserWarning, stacklevel=2)

    def set_session_name(self, name: str = "session"):
        """Set the name for the session.

        The `name` is used to name the session directory, e.g.
        `session_20201208_132800/`. The date is suffixed automatically.
        """
        now = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
        self.session_name = f"{name}_{now}"

    @property
    def session_dir(self):
        """Return session directory."""
        return self["output_dir"] / self.session_name

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
        """Return user config directory.

        .. deprecated:: 2.12.0
            This attribute has been deprecated in ESMValCore version 2.12.0 and
            is scheduled for removal in version 2.14.0.

        """
        msg = (
            "The attribute `Session.config_dir` has been deprecated in "
            "ESMValCore version 2.12.0 and is scheduled for removal in "
            "version 2.14.0."
        )
        warnings.warn(msg, ESMValCoreDeprecationWarning, stacklevel=2)
        if self.get("config_file") is None:
            return None
        return Path(self["config_file"]).parent

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

    clear = ValidatedConfig.clear  # to show this is in API doc

    context = ValidatedConfig.context  # to show this is in API doc

    copy = ValidatedConfig.copy  # to show this is in API doc


def _get_all_config_dirs(cli_config_dir: Path | None) -> list[Path]:
    """Get all configuration directories."""
    config_dirs: list[Path] = [
        DEFAULT_CONFIG_DIR,
        USER_CONFIG_DIR,
    ]
    if cli_config_dir is not None:
        config_dirs.append(cli_config_dir)
    return config_dirs


def _get_all_config_sources(cli_config_dir: Path | None) -> list[str]:
    """Get all sources of configuration directories."""
    config_sources: list[str] = [
        "defaults",
        USER_CONFIG_SOURCE,
    ]
    if cli_config_dir is not None:
        config_sources.append("command line argument")
    return config_sources


def _get_global_config() -> Config:
    """Get global configuration object."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Do not instantiate `Config` objects directly",
            category=UserWarning,
            module="esmvalcore",
        )
        config_obj = Config()
    config_obj.reload()
    return config_obj


# Initialize configuration objects
CFG = _get_global_config()
