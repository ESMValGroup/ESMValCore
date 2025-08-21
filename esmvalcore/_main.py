"""Earth System Model Evaluation Tool

A community tool for the evaluation of Earth system models.

https://esmvaltool.org

The Earth System Model Evaluation Tool (ESMValTool) is a community
diagnostics and performance metrics tool for the evaluation of Earth
System Models (ESMs) that allows for routine comparison of single or
multiple models, either against predecessor versions or against
observations.

Tutorial: https://tutorial.esmvaltool.org
Documentation: https://docs.esmvaltool.org
Contact: esmvaltool-dev@listserv.dfn.de

If you find this software useful for your research, please cite it using
https://doi.org/10.5281/zenodo.3387139 for ESMValCore or
https://doi.org/10.5281/zenodo.3401363 for ESMValTool or
any of the reference papers listed at https://esmvaltool.org/references/.

Have fun!
"""  # noqa: D400

# Imports outside the top level are used here to avoid importing the
# entire package just to print a help message. This makes the command line
# much more responsive.
# ruff: noqa: PLC0415
# pylint: disable=import-outside-toplevel
from __future__ import annotations

import logging
import os
import sys
import warnings
from importlib.metadata import entry_points
from pathlib import Path

import fire

from esmvalcore.config._config import warn_if_old_extra_facets_exist
from esmvalcore.exceptions import ESMValCoreDeprecationWarning

# set up logging
logger = logging.getLogger(__name__)

HEADER = (
    r"""
______________________________________________________________________
          _____ ____  __  ____     __    _ _____           _
         | ____/ ___||  \/  \ \   / /_ _| |_   _|__   ___ | |
         |  _| \___ \| |\/| |\ \ / / _` | | | |/ _ \ / _ \| |
         | |___ ___) | |  | | \ V / (_| | | | | (_) | (_) | |
         |_____|____/|_|  |_|  \_/ \__,_|_| |_|\___/ \___/|_|
______________________________________________________________________

"""
    + __doc__
)


def parse_resume(resume, recipe):
    """Set `resume` to a correct value and sanity check."""
    if not resume:
        return []
    if isinstance(resume, str):
        resume = resume.split(" ")
    for i, resume_dir in enumerate(resume):
        resume[i] = Path(os.path.expandvars(resume_dir)).expanduser()

    # Sanity check resume directories:
    current_recipe = recipe.read_text(encoding="utf-8")
    for resume_dir in resume:
        resume_recipe = resume_dir / "run" / recipe.name
        if current_recipe != resume_recipe.read_text(encoding="utf-8"):
            msg = (
                f"Only identical recipes can be resumed, but "
                f"{resume_recipe} is different from {recipe}"
            )
            raise ValueError(msg)
    return resume


def process_recipe(recipe_file: Path, session):
    """Process recipe."""
    import datetime
    import shutil

    from esmvalcore._recipe.recipe import read_recipe_file

    if not recipe_file.is_file():
        import errno

        raise OSError(
            errno.ENOENT,
            "Specified recipe file does not exist",
            recipe_file,
        )

    timestamp1 = datetime.datetime.now(datetime.UTC)
    timestamp_format = "%Y-%m-%d %H:%M:%S"

    logger.info(
        "Starting the Earth System Model Evaluation Tool at time: %s UTC",
        timestamp1.strftime(timestamp_format),
    )

    logger.info(70 * "-")
    logger.info("RECIPE   = %s", recipe_file)
    logger.info("RUNDIR     = %s", session.run_dir)
    logger.info("WORKDIR    = %s", session.work_dir)
    logger.info("PREPROCDIR = %s", session.preproc_dir)
    logger.info("PLOTDIR    = %s", session.plot_dir)
    logger.info(70 * "-")

    n_processes = session["max_parallel_tasks"] or os.cpu_count()
    logger.info("Running tasks using at most %s processes", n_processes)

    logger.info(
        "If your system hangs during execution, it may not have enough "
        "memory for keeping this number of tasks in memory.",
    )
    logger.info(
        "If you experience memory problems, try reducing "
        "'max_parallel_tasks' in your configuration.",
    )

    if session["compress_netcdf"]:
        logger.warning(
            "You have enabled NetCDF compression. Accessing .nc files can be "
            "much slower than expected if your access pattern does not match "
            "their internal pattern. Make sure to specify the expected "
            "access pattern in the recipe as a parameter to the 'save' "
            "preprocessor function. If the problem persists, try disabling "
            "NetCDF compression.",
        )

    # copy recipe to run_dir for future reference
    shutil.copy2(recipe_file, session.run_dir)

    # parse recipe
    recipe = read_recipe_file(recipe_file, session)
    logger.debug("Recipe summary:\n%s", recipe)
    # run
    recipe.run()
    # End time timing
    timestamp2 = datetime.datetime.now(datetime.UTC)
    logger.info(
        "Ending the Earth System Model Evaluation Tool at time: %s UTC",
        timestamp2.strftime(timestamp_format),
    )
    logger.info("Time for running the recipe was: %s", timestamp2 - timestamp1)


class Config:
    """Manage ESMValTool's configuration.

    This group contains utilities to manage ESMValTool configuration
    files.
    """

    @staticmethod
    def _copy_config_file(
        in_file: Path,
        out_file: Path,
        overwrite: bool,
    ):
        """Copy a configuration file."""
        import shutil

        from .config._logging import configure_logging

        configure_logging(console_log_level="info")

        if out_file.is_file():
            if overwrite:
                logger.info("Overwriting file %s.", out_file)
            else:
                logger.info("Copy aborted. File %s already exists.", out_file)
                return

        target_folder = out_file.parent
        if not target_folder.is_dir():
            logger.info("Creating folder %s", target_folder)
            target_folder.mkdir(parents=True, exist_ok=True)

        logger.info("Copying file %s to path %s.", in_file, out_file)
        shutil.copy2(in_file, out_file)
        logger.info("Copy finished.")

    @classmethod
    def get_config_user(
        cls,
        overwrite: bool = False,
        path: str | Path | None = None,
    ) -> None:
        """Copy default configuration to a given path.

        Copy default configuration to a given path or, if a `path` is not
        provided, install it in the default `~/.config/esmvaltool/` directory.

        Parameters
        ----------
        overwrite:
            Overwrite an existing file.
        path:
            If not provided, the file will be copied to
            `~/.config/esmvaltool/`.

        """
        from .config._config_object import DEFAULT_CONFIG_DIR

        in_file = DEFAULT_CONFIG_DIR / "config-user.yml"
        if path is None:
            out_file = (
                Path.home() / ".config" / "esmvaltool" / "config-user.yml"
            )
        else:
            out_file = Path(path)
        if not out_file.suffix:  # out_file looks like a directory
            out_file = out_file / "config-user.yml"
        cls._copy_config_file(in_file, out_file, overwrite)

    @classmethod
    def get_config_developer(
        cls,
        overwrite: bool = False,
        path: str | Path | None = None,
    ) -> None:
        """Copy default config-developer.yml file to a given path.

        Copy default config-developer.yml file to a given path or, if a path is
        not provided, install it in the default `~/.esmvaltool` folder.

        Parameters
        ----------
        overwrite: boolean
            Overwrite an existing file.
        path: str
            If not provided, the file will be copied to `~/.esmvaltool`.

        """
        in_file = Path(__file__).parent / "config-developer.yml"
        if path is None:
            out_file = Path.home() / ".esmvaltool" / "config-developer.yml"
        else:
            out_file = Path(path)
        if not out_file.suffix:  # out_file looks like a directory
            out_file = out_file / "config-developer.yml"
        cls._copy_config_file(in_file, out_file, overwrite)


class Recipes:
    """List, show and retrieve installed recipes.

    This group contains utilities to explore and manage the recipes available
    in your installation of ESMValTool.

    Documentation for recipes included with ESMValTool is available at
    https://docs.esmvaltool.org/en/latest/recipes/index.html.
    """

    @staticmethod
    def list() -> None:
        """List all installed recipes.

        Show all installed recipes, grouped by folder.
        """
        from .config._diagnostics import DIAGNOSTICS
        from .config._logging import configure_logging

        configure_logging(console_log_level="info")
        recipes_folder = DIAGNOSTICS.recipes
        logger.info("Showing recipes installed in %s", recipes_folder)
        print("# Installed recipes")  # noqa: T201
        for recipe_root, _, files in sorted(os.walk(recipes_folder)):
            root = os.path.relpath(recipe_root, recipes_folder)
            if root == ".":
                root = ""
            if root:
                print(f"\n# {root.replace(os.sep, ' - ').title()}")  # noqa: T201
            for filename in sorted(files):
                if filename.endswith(".yml"):
                    print(os.path.join(root, filename))  # noqa: T201

    @staticmethod
    def get(recipe: str) -> None:
        """Get a copy of any installed recipe in the current working directory.

        Use this command to get a local copy of any installed recipe.

        Parameters
        ----------
        recipe: str
            Name of the recipe to get, including any subdirectories.
        """
        import shutil

        from .config._diagnostics import DIAGNOSTICS
        from .config._logging import configure_logging
        from .exceptions import RecipeError

        configure_logging(console_log_level="info")
        installed_recipe = DIAGNOSTICS.recipes / recipe
        if not installed_recipe.exists():
            msg = (
                f"Recipe {recipe} not found. To list all available recipes, "
                'execute "esmvaltool list"'
            )
            raise RecipeError(msg)
        logger.info("Copying installed recipe to the current folder...")
        shutil.copy(installed_recipe, Path(recipe).name)
        logger.info("Recipe %s successfully copied", recipe)

    @staticmethod
    def show(recipe: str) -> None:
        """Show the given recipe in console.

        Use this command to see the contents of any installed recipe.

        Parameters
        ----------
        recipe: str
            Name of the recipe to get, including any subdirectories.
        """
        from .config._diagnostics import DIAGNOSTICS
        from .config._logging import configure_logging
        from .exceptions import RecipeError

        configure_logging(console_log_level="info")
        installed_recipe = DIAGNOSTICS.recipes / recipe
        if not installed_recipe.exists():
            msg = (
                f"Recipe {recipe} not found. To list all available recipes, "
                'execute "esmvaltool list"'
            )
            raise RecipeError(msg)
        msg = f"Recipe {recipe}"
        logger.info(msg)
        logger.info("=" * len(msg))
        print(installed_recipe.read_text(encoding="utf-8"))  # noqa: T201


class ESMValTool:
    # This is the `esmvaltool` command. The line below shows the documentation
    # at the top of this module when users run e.g. `esmvaltool -- --help`.
    __doc__ = __doc__

    def __init__(self):
        self.config = Config()
        self.recipes = Recipes()
        self._extra_packages = {}
        esmvaltool_commands = entry_points(group="esmvaltool_commands")
        if not esmvaltool_commands:
            print(  # noqa: T201
                "Running esmvaltool executable from ESMValCore. "
                "No other command line utilities are available "
                "until ESMValTool is installed.",
            )
        for entry_point in esmvaltool_commands:
            self._extra_packages[entry_point.dist.name] = (
                entry_point.dist.version
            )
            if hasattr(self, entry_point.name):
                logger.error(
                    "Registered command %s already exists",
                    entry_point.name,
                )
                continue
            self.__setattr__(entry_point.name, entry_point.load()())

    def version(self):
        """Show versions of all packages that form ESMValTool.

        In particular, this command will show the version ESMValCore and
        any other package that adds a subcommand to 'esmvaltool'
        command.
        """
        from . import __version__

        print(f"ESMValCore: {__version__}")  # noqa: T201
        for project, version in self._extra_packages.items():
            print(f"{project}: {version}")  # noqa: T201

    def run(self, recipe, **kwargs):
        """Execute an ESMValTool recipe.

        `esmvaltool run` executes the given recipe. To see a list of available
        recipes or create a local copy of any of them, use the
        `esmvaltool recipes` command group.

        A list of possible flags is given here:
        https://docs.esmvaltool.org/projects/ESMValCore/en/latest/quickstart/configure.html#configuration-options

        """
        from .config import CFG
        from .config._dask import warn_if_old_dask_config_exists
        from .exceptions import InvalidConfigParameter

        cli_config_dir = kwargs.pop("config_dir", None)
        if cli_config_dir is not None:
            cli_config_dir = Path(cli_config_dir).expanduser().absolute()
            if not cli_config_dir.is_dir():
                msg = (
                    f"Invalid --config_dir given: {cli_config_dir} is not an "
                    f"existing directory"
                )
                raise NotADirectoryError(msg)

        # TODO: remove in v2.14.0
        # At this point, --config_file is already parsed if a valid file has
        # been given (see
        # https://github.com/ESMValGroup/ESMValCore/issues/2280), but no error
        # has been raised if the file does not exist. Thus, reload the file
        # here with `load_from_file` to make sure a proper error is raised.
        if "config_file" in kwargs:
            if os.environ.get("ESMVALTOOL_CONFIG_DIR"):
                deprecation_msg = (
                    "Usage of a single configuration file specified via CLI "
                    "argument `--config_file` has been deprecated in "
                    "ESMValCore version 2.12.0 and is scheduled for removal "
                    "in version 2.14.0. Since the environment variable "
                    "ESMVALTOOL_CONFIG_DIR is set, old configuration files "
                    "present at ~/.esmvaltool/config-user.yml and/or "
                    "specified via `--config_file` are currently ignored. To "
                    "silence this warning, omit CLI argument `--config_file`."
                )
                warnings.warn(
                    deprecation_msg,
                    ESMValCoreDeprecationWarning,
                    stacklevel=2,
                )
                kwargs.pop("config_file")
            else:
                cli_config_dir = kwargs["config_file"]
                CFG.load_from_file(kwargs["config_file"])

        # New in v2.12.0: read additional configuration directory given by CLI
        # argument
        if CFG.get("config_file") is None and cli_config_dir is not None:
            try:
                CFG.update_from_dirs([cli_config_dir])
            except InvalidConfigParameter as exc:
                msg = (
                    f"Failed to parse configuration directory "
                    f"{cli_config_dir} (command line argument): "
                    f"{exc!s}"
                )
                raise InvalidConfigParameter(msg) from exc

        recipe = self._get_recipe(recipe)

        # Parse command line arguments
        try:
            CFG.nested_update(kwargs)
        except InvalidConfigParameter as exc:
            msg = f"Invalid command line argument given: {exc!s}"
            raise InvalidConfigParameter(msg) from exc

        CFG["resume_from"] = parse_resume(CFG["resume_from"], recipe)
        session = CFG.start_session(recipe.stem)

        self._run(recipe, session, cli_config_dir)

        # Print warnings about deprecated configuration options again
        CFG.reload()
        if cli_config_dir is not None:
            CFG.update_from_dirs([cli_config_dir])
        CFG.nested_update(kwargs)

        warn_if_old_dask_config_exists()
        warn_if_old_extra_facets_exist()

    @staticmethod
    def _create_session_dir(session):
        """Create `session.session_dir` or an alternative if it exists."""
        from .exceptions import RecipeError

        session_dir = session.session_dir
        for suffix in range(1, 1000):
            try:
                session_dir.mkdir(parents=True)
            except FileExistsError:
                session_dir = Path(f"{session.session_dir}-{suffix}")
            else:
                session.session_name = session_dir.name
                return

        msg = (
            f"Output directory '{session.session_dir}' already exists and"
            " unable to find alternative, aborting to prevent data loss."
        )
        raise RecipeError(msg)

    def _run(
        self,
        recipe: Path,
        session,
        cli_config_dir: Path | None,
    ) -> None:
        """Run `recipe` using `session`."""
        self._create_session_dir(session)
        session.run_dir.mkdir()

        # configure logging
        from .config._logging import configure_logging

        log_files = configure_logging(
            output_dir=session.run_dir,
            console_log_level=session["log_level"],
        )
        self._log_header(log_files, cli_config_dir)

        # configure resource logger and run program
        from ._task import resource_usage_logger

        resource_log = session.run_dir / "resource_usage.txt"
        with resource_usage_logger(pid=os.getpid(), filename=resource_log):
            process_recipe(recipe_file=recipe, session=session)

        self._clean_preproc(session)

        if session.cmor_log.read_text(encoding="utf-8"):
            logger.warning(
                "Input data is not (fully) CMOR-compliant, see %s for details",
                session.cmor_log,
            )

        logger.info("Run was successful")

    @staticmethod
    def _clean_preproc(session):
        import shutil

        if (
            not session["save_intermediary_cubes"]
            and session._fixed_file_dir.exists()  # noqa: SLF001
        ):
            logger.debug(
                "Removing `preproc/fixed_files` directory containing fixed "
                "data",
            )
            logger.debug(
                "If this data is further needed, then set "
                "`save_intermediary_cubes` to `true` and `remove_preproc_dir` "
                "to `false` in your configuration",
            )
            shutil.rmtree(session._fixed_file_dir)  # noqa: SLF001

        if session["remove_preproc_dir"] and session.preproc_dir.exists():
            logger.info(
                "Removing `preproc` directory containing preprocessed data",
            )
            logger.info(
                "If this data is further needed, then set "
                "`remove_preproc_dir` to `false` in your configuration",
            )
            shutil.rmtree(session.preproc_dir)

    @staticmethod
    def _get_recipe(recipe) -> Path:
        from esmvalcore.config._diagnostics import DIAGNOSTICS

        if not os.path.isfile(recipe):
            installed_recipe = DIAGNOSTICS.recipes / recipe
            if os.path.isfile(installed_recipe):
                recipe = installed_recipe
        return Path(os.path.expandvars(recipe)).expanduser().absolute()

    @staticmethod
    def _get_config_info(cli_config_dir):
        """Get information about config files for logging."""
        from .config import CFG
        from .config._config_object import (
            DEFAULT_CONFIG_DIR,
            _get_all_config_dirs,
            _get_all_config_sources,
        )

        # TODO: remove in v2.14.0
        if CFG.get("config_file") is not None:
            config_info = [
                (DEFAULT_CONFIG_DIR, "defaults"),
                (CFG["config_file"], "single configuration file [deprecated]"),
            ]

        # New in v2.12.0
        else:
            config_dirs = []
            for path in _get_all_config_dirs(cli_config_dir):
                if not path.is_dir():
                    config_dirs.append(f"{path} [NOT AN EXISTING DIRECTORY]")
                else:
                    config_dirs.append(str(path))
            config_info = list(
                zip(
                    config_dirs,
                    _get_all_config_sources(cli_config_dir),
                    strict=False,
                ),
            )

        return "\n".join(f"{i[0]} ({i[1]})" for i in config_info)

    def _log_header(self, log_files, cli_config_dir):
        from . import __version__

        logger.info(HEADER)
        logger.info("Package versions")
        logger.info("----------------")
        logger.info("ESMValCore: %s", __version__)
        for project, version in self._extra_packages.items():
            logger.info("%s: %s", project, version)
        logger.info("----------------")
        logger.info(
            "Reading configuration files from:\n%s",
            self._get_config_info(cli_config_dir),
        )
        logger.info("Writing program log files to:\n%s", "\n".join(log_files))


def run():
    """Run the `esmvaltool` program, logging any exceptions."""
    from .exceptions import RecipeError

    # Workaround to avoid using more for the output

    def display(lines, out):
        text = "\n".join(lines) + "\n"
        out.write(text)

    fire.core.Display = display

    try:
        fire.Fire(ESMValTool())
    except fire.core.FireExit:
        raise
    except RecipeError as exc:
        # Hide the stack trace for RecipeErrors
        logger.error("%s", exc)
        logger.debug("Stack trace for debugging:", exc_info=True)
        sys.exit(1)
    except Exception:
        if not logger.handlers:
            # Add a logging handler if main failed to do so.
            logging.basicConfig()
        logger.exception(
            "Program terminated abnormally, see stack trace "
            "below for more information:",
        )
        logger.info(
            "\n"
            "If you have a question or need help, please start a new "
            "discussion on "
            "https://github.com/ESMValGroup/ESMValTool/discussions"
            "\n"
            "If you suspect this is a bug, please open an issue on "
            "https://github.com/ESMValGroup/ESMValTool/issues"
            "\n"
            "To make it easier to find out what the problem is, please "
            "consider attaching the files run/recipe_*.yml and "
            "run/main_log_debug.txt from the output directory.",
        )
        sys.exit(1)
