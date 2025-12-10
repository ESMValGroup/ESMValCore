"""Importable config object."""

from __future__ import annotations

import datetime
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import dask.config

import esmvalcore
from esmvalcore.config._config_validators import (
    _deprecated_options_defaults,
    _deprecators,
    _validators,
)
from esmvalcore.config._validated_config import ValidatedConfig
from esmvalcore.exceptions import InvalidConfigParameter

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
            raise NotADirectoryError(msg)
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

    _validate = _validators
    _deprecate = _deprecators
    _deprecated_defaults = _deprecated_options_defaults
    _warn_if_missing = (("projects", URL),)

    def __init__(self, *args, **kwargs):
        """Initialize class instance."""
        super().__init__(*args, **kwargs)
        msg = (
            "Do not instantiate `Config` objects directly, this will lead "
            "to unexpected behavior. Use `esmvalcore.config.CFG` instead."
        )
        warnings.warn(msg, UserWarning, stacklevel=2)

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

    def __init__(self, config: dict, name: str = "session") -> None:
        super().__init__(config)
        self.session_name: str | None = None
        self.set_session_name(name)
        msg = (
            "Do not instantiate `Session` objects directly, this will lead "
            "to unexpected behavior. Use "
            "`esmvalcore.config.CFG.start_session` instead."
        )
        warnings.warn(msg, UserWarning, stacklevel=2)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(<config>, name={self.session_name})"

    def __str__(self) -> str:
        return repr(self)

    def set_session_name(self, name: str = "session") -> None:
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
