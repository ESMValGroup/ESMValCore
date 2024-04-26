"""ESMValTool - Earth System Model Evaluation Tool.

http://www.esmvaltool.org

CORE DEVELOPMENT TEAM AND CONTACTS:
  Birgit Hassler (Co-PI; DLR, Germany - birgit.hassler@dlr.de)
  Alistair Sellar (Co-PI; Met Office, UK - alistair.sellar@metoffice.gov.uk)
  Bouwe Andela (Netherlands eScience Center, The Netherlands - b.andela@esciencecenter.nl)
  Lee de Mora (PML, UK - ledm@pml.ac.uk)
  Niels Drost (Netherlands eScience Center, The Netherlands - n.drost@esciencecenter.nl)
  Veronika Eyring (DLR, Germany - veronika.eyring@dlr.de)
  Bettina Gier (UBremen, Germany - gier@uni-bremen.de)
  Remi Kazeroni (DLR, Germany - remi.kazeroni@dlr.de)
  Nikolay Koldunov (AWI, Germany - nikolay.koldunov@awi.de)
  Axel Lauer (DLR, Germany - axel.lauer@dlr.de)
  Saskia Loosveldt-Tomas (BSC, Spain - saskia.loosveldt@bsc.es)
  Ruth Lorenz (ETH Zurich, Switzerland - ruth.lorenz@env.ethz.ch)
  Benjamin Mueller (LMU, Germany - b.mueller@iggf.geo.uni-muenchen.de)
  Valeriu Predoi (URead, UK - valeriu.predoi@ncas.ac.uk)
  Mattia Righi (DLR, Germany - mattia.righi@dlr.de)
  Manuel Schlund (DLR, Germany - manuel.schlund@dlr.de)
  Breixo Solino Fernandez (DLR, Germany - breixo.solinofernandez@dlr.de)
  Javier Vegas-Regidor (BSC, Spain - javier.vegas@bsc.es)
  Klaus Zimmermann (SMHI, Sweden - klaus.zimmermann@smhi.se)

For further help, please read the documentation at
http://docs.esmvaltool.org. Have fun!
"""  # noqa: line-too-long pylint: disable=line-too-long
# pylint: disable=import-outside-toplevel
import logging
import os
import sys
from pathlib import Path

if (sys.version_info.major, sys.version_info.minor) < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points  # type: ignore

import fire

# set up logging
logger = logging.getLogger(__name__)

HEADER = r"""
______________________________________________________________________
          _____ ____  __  ____     __    _ _____           _
         | ____/ ___||  \/  \ \   / /_ _| |_   _|__   ___ | |
         |  _| \___ \| |\/| |\ \ / / _` | | | |/ _ \ / _ \| |
         | |___ ___) | |  | | \ V / (_| | | | | (_) | (_) | |
         |_____|____/|_|  |_|  \_/ \__,_|_| |_|\___/ \___/|_|
______________________________________________________________________

""" + __doc__


def parse_resume(resume, recipe):
    """Set `resume` to a correct value and sanity check."""
    if not resume:
        return []
    if isinstance(resume, str):
        resume = resume.split(' ')
    for i, resume_dir in enumerate(resume):
        resume[i] = Path(os.path.expandvars(resume_dir)).expanduser()

    # Sanity check resume directories:
    current_recipe = recipe.read_text(encoding='utf-8')
    for resume_dir in resume:
        resume_recipe = resume_dir / 'run' / recipe.name
        if current_recipe != resume_recipe.read_text(encoding='utf-8'):
            raise ValueError(f'Only identical recipes can be resumed, but '
                             f'{resume_recipe} is different from {recipe}')
    return resume


def process_recipe(recipe_file: Path, session):
    """Process recipe."""
    import datetime
    import shutil

    from esmvalcore._recipe.recipe import read_recipe_file
    from esmvalcore.config._dask import check_distributed_config
    if not recipe_file.is_file():
        import errno
        raise OSError(errno.ENOENT, "Specified recipe file does not exist",
                      recipe_file)

    timestamp1 = datetime.datetime.utcnow()
    timestamp_format = "%Y-%m-%d %H:%M:%S"

    logger.info(
        "Starting the Earth System Model Evaluation Tool at time: %s UTC",
        timestamp1.strftime(timestamp_format))

    logger.info(70 * "-")
    logger.info("RECIPE   = %s", recipe_file)
    logger.info("RUNDIR     = %s", session.run_dir)
    logger.info("WORKDIR    = %s", session.work_dir)
    logger.info("PREPROCDIR = %s", session.preproc_dir)
    logger.info("PLOTDIR    = %s", session.plot_dir)
    logger.info(70 * "-")

    n_processes = session['max_parallel_tasks'] or os.cpu_count()
    logger.info("Running tasks using at most %s processes", n_processes)

    logger.info(
        "If your system hangs during execution, it may not have enough "
        "memory for keeping this number of tasks in memory.")
    logger.info("If you experience memory problems, try reducing "
                "'max_parallel_tasks' in your user configuration file.")

    check_distributed_config()

    if session['compress_netcdf']:
        logger.warning(
            "You have enabled NetCDF compression. Accessing .nc files can be "
            "much slower than expected if your access pattern does not match "
            "their internal pattern. Make sure to specify the expected "
            "access pattern in the recipe as a parameter to the 'save' "
            "preprocessor function. If the problem persists, try disabling "
            "NetCDF compression.")

    # copy recipe to run_dir for future reference
    shutil.copy2(recipe_file, session.run_dir)

    # parse recipe
    recipe = read_recipe_file(recipe_file, session)
    logger.debug("Recipe summary:\n%s", recipe)
    # run
    recipe.run()
    # End time timing
    timestamp2 = datetime.datetime.utcnow()
    logger.info(
        "Ending the Earth System Model Evaluation Tool at time: %s UTC",
        timestamp2.strftime(timestamp_format))
    logger.info("Time for running the recipe was: %s", timestamp2 - timestamp1)


class Config():
    """Manage ESMValTool's configuration.

    This group contains utilities to manage ESMValTool configuration
    files.
    """

    @staticmethod
    def _copy_config_file(filename, overwrite, path):
        import shutil

        from .config._logging import configure_logging
        configure_logging(console_log_level='info')
        if not path:
            path = os.path.join(os.path.expanduser('~/.esmvaltool'), filename)
        if os.path.isfile(path):
            if overwrite:
                logger.info('Overwriting file %s.', path)
            else:
                logger.info('Copy aborted. File %s already exists.', path)
                return

        target_folder = os.path.dirname(path)
        if not os.path.isdir(target_folder):
            logger.info('Creating folder %s', target_folder)
            os.makedirs(target_folder)

        conf_file = os.path.join(os.path.dirname(__file__), filename)
        logger.info('Copying file %s to path %s.', conf_file, path)
        shutil.copy2(conf_file, path)
        logger.info('Copy finished.')

    @classmethod
    def get_config_user(cls, overwrite=False, path=None):
        """Copy default config-user.yml file to a given path.

        Copy default config-user.yml file to a given path or, if a path is
        not provided, install it in the default `${HOME}/.esmvaltool` folder.

        Parameters
        ----------
        overwrite: boolean
            Overwrite an existing file.
        path: str
            If not provided, the file will be copied to
            .esmvaltool in the user's home.
        """
        cls._copy_config_file('config-user.yml', overwrite, path)

    @classmethod
    def get_config_developer(cls, overwrite=False, path=None):
        """Copy default config-developer.yml file to a given path.

        Copy default config-developer.yml file to a given path or, if a path is
        not provided, install it in the default `${HOME}/.esmvaltool` folder.

        Parameters
        ----------
        overwrite: boolean
            Overwrite an existing file.
        path: str
            If not provided, the file will be copied to
            .esmvaltool in the user's home.
        """
        cls._copy_config_file('config-developer.yml', overwrite, path)


class Recipes():
    """List, show and retrieve installed recipes.

    This group contains utilities to explore and manage the recipes available
    in your installation of ESMValTool.

    Documentation for recipes included with ESMValTool is available at
    https://docs.esmvaltool.org/en/latest/recipes/index.html.
    """

    @staticmethod
    def list():
        """List all installed recipes.

        Show all installed recipes, grouped by folder.
        """
        from .config._diagnostics import DIAGNOSTICS
        from .config._logging import configure_logging
        configure_logging(console_log_level='info')
        recipes_folder = DIAGNOSTICS.recipes
        logger.info("Showing recipes installed in %s", recipes_folder)
        print('# Installed recipes')
        for root, _, files in sorted(os.walk(recipes_folder)):
            root = os.path.relpath(root, recipes_folder)
            if root == '.':
                root = ''
            if root:
                print(f"\n# {root.replace(os.sep, ' - ').title()}")
            for filename in sorted(files):
                if filename.endswith('.yml'):
                    print(os.path.join(root, filename))

    @staticmethod
    def get(recipe):
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
        configure_logging(console_log_level='info')
        installed_recipe = DIAGNOSTICS.recipes / recipe
        if not installed_recipe.exists():
            ValueError(
                f'Recipe {recipe} not found. To list all available recipes, '
                'execute "esmvaltool list"')
        logger.info('Copying installed recipe to the current folder...')
        shutil.copy(installed_recipe, Path(recipe).name)
        logger.info('Recipe %s successfully copied', recipe)

    @staticmethod
    def show(recipe):
        """Show the given recipe in console.

        Use this command to see the contents of any installed recipe.

        Parameters
        ----------
        recipe: str
            Name of the recipe to get, including any subdirectories.
        """
        from .config._diagnostics import DIAGNOSTICS
        from .config._logging import configure_logging
        configure_logging(console_log_level='info')
        installed_recipe = DIAGNOSTICS.recipes / recipe
        if not installed_recipe.exists():
            ValueError(
                f'Recipe {recipe} not found. To list all available recipes, '
                'execute "esmvaltool list"')
        msg = f'Recipe {recipe}'
        logger.info(msg)
        logger.info('=' * len(msg))
        print(installed_recipe.read_text(encoding='utf-8'))


class ESMValTool():
    """A community tool for routine evaluation of Earth system models.

    The Earth System Model Evaluation Tool (ESMValTool) is a community
    diagnostics and performance metrics tool for the evaluation of Earth
    System Models (ESMs) that allows for routine comparison of single or
    multiple models, either against predecessor versions or against
    observations.

    Documentation is available at https://docs.esmvaltool.org.

    To report issues or ask for improvements, please visit
    https://github.com/ESMValGroup/ESMValTool.
    """

    def __init__(self):
        self.config = Config()
        self.recipes = Recipes()
        self._extra_packages = {}
        esmvaltool_commands = entry_points(group='esmvaltool_commands')
        if not esmvaltool_commands:
            print("Running esmvaltool executable from ESMValCore. "
                  "No other command line utilities are available "
                  "until ESMValTool is installed.")
        for entry_point in esmvaltool_commands:
            self._extra_packages[entry_point.dist.name] = \
                entry_point.dist.version
            if hasattr(self, entry_point.name):
                logger.error('Registered command %s already exists',
                             entry_point.name)
                continue
            self.__setattr__(entry_point.name, entry_point.load()())

    def version(self):
        """Show versions of all packages that form ESMValTool.

        In particular, this command will show the version ESMValCore and
        any other package that adds a subcommand to 'esmvaltool'
        command.
        """
        from . import __version__
        print(f'ESMValCore: {__version__}')
        for project, version in self._extra_packages.items():
            print(f'{project}: {version}')

    def run(self,
            recipe,
            config_file=None,
            resume_from=None,
            max_datasets=None,
            max_years=None,
            skip_nonexistent=None,
            search_esgf=None,
            diagnostics=None,
            check_level=None,
            **kwargs):
        """Execute an ESMValTool recipe.

        `esmvaltool run` executes the given recipe. To see a list of available
        recipes or create a local copy of any of them, use the
        `esmvaltool recipes` command group.

        Parameters
        ----------
        recipe : str
            Recipe to run, as either the name of an installed recipe or the
            path to a non-installed one.
        config_file: str, optional
            Configuration file to use. Can be given as absolute or relative
            path. In the latter case, search in the current working directory
            and `${HOME}/.esmvaltool` (in that order). If not provided, the
            file `${HOME}/.esmvaltool/config-user.yml` will be used.
        resume_from: list(str), optional
            Resume one or more previous runs by using preprocessor output files
            from these output directories.
        max_datasets: int, optional
            Maximum number of datasets to use.
        max_years: int, optional
            Maximum number of years to use.
        skip_nonexistent: bool, optional
            If True, the run will not fail if some datasets are not available.
        search_esgf: str, optional
            If `never`, disable automatic download of data from the ESGF. If
            `when_missing`, enable the automatic download of files that are not
            available locally. If `always`, always check ESGF for the latest
            version of a file, and only use local files if they correspond to
            that latest version.
        diagnostics: list(str), optional
            Only run the selected diagnostics from the recipe. To provide more
            than one diagnostic to filter use the syntax 'diag1 diag2/script1'
            or '("diag1", "diag2/script1")' and pay attention to the quotes.
        check_level: str, optional
            Configure the sensitivity of the CMOR check. Possible values are:
            `ignore` (all errors will be reported as warnings),
            `relaxed` (only fail if there are critical errors),
            default (fail if there are any errors),
            strict (fail if there are any warnings).
        """
        from .config import CFG

        # At this point, --config_file is already parsed if a valid file has
        # been given (see
        # https://github.com/ESMValGroup/ESMValCore/issues/2280), but no error
        # has been raised if the file does not exist. Thus, reload the file
        # here with `load_from_file` to make sure a proper error is raised.
        CFG.load_from_file(config_file)

        recipe = self._get_recipe(recipe)

        session = CFG.start_session(recipe.stem)
        if check_level is not None:
            session['check_level'] = check_level
        if diagnostics is not None:
            session['diagnostics'] = diagnostics
        if max_datasets is not None:
            session['max_datasets'] = max_datasets
        if max_years is not None:
            session['max_years'] = max_years
        if search_esgf is not None:
            session['search_esgf'] = search_esgf
        if skip_nonexistent is not None:
            session['skip_nonexistent'] = skip_nonexistent
        session['resume_from'] = parse_resume(resume_from, recipe)
        session.update(kwargs)

        self._run(recipe, session)
        # Print warnings about deprecated configuration options again:
        CFG.reload()

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

        raise RecipeError(
            f"Output directory '{session.session_dir}' already exists and"
            " unable to find alternative, aborting to prevent data loss.")

    def _run(self, recipe: Path, session) -> None:
        """Run `recipe` using `session`."""
        self._create_session_dir(session)
        session.run_dir.mkdir()

        # configure logging
        from .config._logging import configure_logging
        log_files = configure_logging(output_dir=session.run_dir,
                                      console_log_level=session['log_level'])
        self._log_header(session['config_file'], log_files)

        if session['search_esgf'] != 'never':
            from .esgf._logon import logon
            logon()

        # configure resource logger and run program
        from ._task import resource_usage_logger
        resource_log = session.run_dir / 'resource_usage.txt'
        with resource_usage_logger(pid=os.getpid(), filename=resource_log):
            process_recipe(recipe_file=recipe, session=session)

        self._clean_preproc(session)

        if session.cmor_log.read_text(encoding='utf-8'):
            logger.warning(
                "Input data is not (fully) CMOR-compliant, see %s for details",
                session.cmor_log,
            )

        logger.info("Run was successful")

    @staticmethod
    def _clean_preproc(session):
        import shutil

        if (not session['save_intermediary_cubes'] and
                session._fixed_file_dir.exists()):
            logger.debug(
                "Removing `preproc/fixed_files` directory containing fixed "
                "data"
            )
            logger.debug(
                "If this data is further needed, then set "
                "`save_intermediary_cubes` to `true` and `remove_preproc_dir` "
                "to `false` in your user configuration file"
            )
            shutil.rmtree(session._fixed_file_dir)

        if session['remove_preproc_dir'] and session.preproc_dir.exists():
            logger.info(
                "Removing `preproc` directory containing preprocessed data"
            )
            logger.info(
                "If this data is further needed, then set "
                "`remove_preproc_dir` to `false` in your user configuration "
                "file"
            )
            shutil.rmtree(session.preproc_dir)

    @staticmethod
    def _get_recipe(recipe) -> Path:
        from esmvalcore.config._diagnostics import DIAGNOSTICS
        if not os.path.isfile(recipe):
            installed_recipe = DIAGNOSTICS.recipes / recipe
            if os.path.isfile(installed_recipe):
                recipe = installed_recipe
        recipe = Path(os.path.expandvars(recipe)).expanduser().absolute()
        return recipe

    def _log_header(self, config_file, log_files):
        from . import __version__
        logger.info(HEADER)
        logger.info('Package versions')
        logger.info('----------------')
        logger.info('ESMValCore: %s', __version__)
        for project, version in self._extra_packages.items():
            logger.info('%s: %s', project, version)
        logger.info('----------------')
        logger.info("Using config file %s", config_file)
        logger.info("Writing program log files to:\n%s", "\n".join(log_files))


def run():
    """Run the `esmvaltool` program, logging any exceptions."""
    import sys

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
    except Exception:  # noqa
        if not logger.handlers:
            # Add a logging handler if main failed to do so.
            logging.basicConfig()
        logger.exception(
            "Program terminated abnormally, see stack trace "
            "below for more information:",
            exc_info=True)
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
            "run/main_log_debug.txt from the output directory.")
        sys.exit(1)
