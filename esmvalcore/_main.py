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
from pathlib import Path

import fire
from pkg_resources import iter_entry_points

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
    import os
    if not resume:
        return []
    if isinstance(resume, str):
        resume = resume.split(' ')
    for i, resume_dir in enumerate(resume):
        resume[i] = Path(os.path.expandvars(resume_dir)).expanduser()

    # Sanity check resume directories:
    current_recipe = recipe.read_text()
    for resume_dir in resume:
        resume_recipe = resume_dir / 'run' / recipe.name
        if current_recipe != resume_recipe.read_text():
            raise ValueError(f'Only identical recipes can be resumed, but '
                             f'{resume_recipe} is different from {recipe}')
    return resume


def process_recipe(recipe_file, config_user):
    """Process recipe."""
    import datetime
    import os
    import shutil

    from ._recipe import read_recipe_file
    if not os.path.isfile(recipe_file):
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
    logger.info("RUNDIR     = %s", config_user['run_dir'])
    logger.info("WORKDIR    = %s", config_user["work_dir"])
    logger.info("PREPROCDIR = %s", config_user["preproc_dir"])
    logger.info("PLOTDIR    = %s", config_user["plot_dir"])
    logger.info(70 * "-")

    from multiprocessing import cpu_count
    n_processes = config_user['max_parallel_tasks'] or cpu_count()
    logger.info("Running tasks using at most %s processes", n_processes)

    logger.info(
        "If your system hangs during execution, it may not have enough "
        "memory for keeping this number of tasks in memory.")
    logger.info("If you experience memory problems, try reducing "
                "'max_parallel_tasks' in your user configuration file.")

    if config_user['compress_netcdf']:
        logger.warning(
            "You have enabled NetCDF compression. Accessing .nc files can be "
            "much slower than expected if your access pattern does not match "
            "their internal pattern. Make sure to specify the expected "
            "access pattern in the recipe as a parameter to the 'save' "
            "preprocessor function. If the problem persists, try disabling "
            "NetCDF compression.")

    # copy recipe to run_dir for future reference
    shutil.copy2(recipe_file, config_user['run_dir'])

    # parse recipe
    recipe = read_recipe_file(recipe_file, config_user)
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
        import os
        import shutil

        from ._config import configure_logging
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
        import os

        from ._config import DIAGNOSTICS, configure_logging
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

        from ._config import DIAGNOSTICS, configure_logging
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
        from ._config import DIAGNOSTICS, configure_logging
        configure_logging(console_log_level='info')
        installed_recipe = DIAGNOSTICS.recipes / recipe
        if not installed_recipe.exists():
            ValueError(
                f'Recipe {recipe} not found. To list all available recipes, '
                'execute "esmvaltool list"')
        msg = f'Recipe {recipe}'
        logger.info(msg)
        logger.info('=' * len(msg))
        with open(installed_recipe) as recipe_file:
            print(recipe_file.read())


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
        self.recipes = Recipes()
        self.config = Config()
        self._extra_packages = {}
        for entry_point in iter_entry_points('esmvaltool_commands'):
            self._extra_packages[entry_point.dist.project_name] = \
                entry_point.dist.version
            if hasattr(self, entry_point.name):
                logger.error('Registered command %s already exists',
                             entry_point.name)
                continue
            self.__setattr__(entry_point.name, entry_point.load()())

    def version(self):
        """Show versions of all packages that conform ESMValTool.

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
            skip_nonexistent=False,
            offline=None,
            diagnostics=None,
            check_level='default',
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
            Configuration file to use. If not provided the file
            ${HOME}/.esmvaltool/config-user.yml will be used.
        resume_from: list(str), optional
            Resume one or more previous runs by using preprocessor output files
            from these output directories.
        max_datasets: int, optional
            Maximum number of datasets to use.
        max_years: int, optional
            Maximum number of years to use.
        skip_nonexistent: bool, optional
            If True, the run will not fail if some datasets are not available.
        offline: bool, optional
            If True, the tool will not download missing data from ESGF.
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
        import os
        import warnings

        from ._config import configure_logging, read_config_user_file
        from ._recipe import TASKSEP
        from .cmor.check import CheckLevels
        from .esgf._logon import logon

        # Check validity of optional command line arguments with experimental
        # API
        with warnings.catch_warnings():
            # ignore experimental API warning
            warnings.simplefilter("ignore")
            from .experimental.config._config_object import Config as ExpConfig
        explicit_optional_kwargs = {
            'config_file': config_file,
            'resume_from': resume_from,
            'max_datasets': max_datasets,
            'max_years': max_years,
            'skip_nonexistent': skip_nonexistent,
            'offline': offline,
            'diagnostics': diagnostics,
            'check_level': check_level,
        }
        all_optional_kwargs = dict(kwargs)
        for (key, val) in explicit_optional_kwargs.items():
            if val is not None:
                all_optional_kwargs[key] = val
        ExpConfig(all_optional_kwargs)

        recipe = self._get_recipe(recipe)
        cfg = read_config_user_file(config_file, recipe.stem, kwargs)

        # Create run dir
        if os.path.exists(cfg['run_dir']):
            print("ERROR: run_dir {} already exists, aborting to "
                  "prevent data loss".format(cfg['output_dir']))
        os.makedirs(cfg['run_dir'])

        # configure logging
        log_files = configure_logging(output_dir=cfg['run_dir'],
                                      console_log_level=cfg['log_level'])

        self._log_header(cfg['config_file'], log_files)

        cfg['resume_from'] = parse_resume(resume_from, recipe)
        cfg['skip_nonexistent'] = skip_nonexistent
        if isinstance(diagnostics, str):
            diagnostics = diagnostics.split(' ')
        cfg['diagnostics'] = {
            pattern if TASKSEP in pattern else pattern + TASKSEP + '*'
            for pattern in diagnostics or ()
        }
        cfg['check_level'] = CheckLevels[check_level.upper()]
        if offline is not None:
            # Override config-user.yml from command line
            cfg['offline'] = offline
        if not cfg['offline']:
            logon()

        def _check_limit(limit, value):
            if value is not None and value < 1:
                raise ValueError("--{} should be larger than 0.".format(
                    limit.replace('_', '-')))
            if value:
                cfg[limit] = value

        _check_limit('max_datasets', max_datasets)
        _check_limit('max_years', max_years)

        resource_log = os.path.join(cfg['run_dir'], 'resource_usage.txt')
        from ._task import resource_usage_logger
        with resource_usage_logger(pid=os.getpid(), filename=resource_log):
            process_recipe(recipe_file=recipe, config_user=cfg)

        self._clean_preproc(cfg)
        logger.info("Run was successful")

    @staticmethod
    def _clean_preproc(cfg):
        import os
        import shutil

        if os.path.exists(cfg["preproc_dir"]) and cfg["remove_preproc_dir"]:
            logger.info("Removing preproc containing preprocessed data")
            logger.info("If this data is further needed, then")
            logger.info("set remove_preproc_dir to false in config-user.yml")
            shutil.rmtree(cfg["preproc_dir"])

    @staticmethod
    def _get_recipe(recipe):
        import os

        from esmvalcore._config import DIAGNOSTICS
        if not os.path.isfile(recipe):
            installed_recipe = str(DIAGNOSTICS.recipes / recipe)
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
