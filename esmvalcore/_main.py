"""ESMValTool - Earth System Model Evaluation Tool.

http://www.esmvaltool.org

CORE DEVELOPMENT TEAM AND CONTACTS:
  Veronika Eyring (PI; DLR, Germany - veronika.eyring@dlr.de)
  Bouwe Andela (NLESC, Netherlands - b.andela@esciencecenter.nl)
  Bjoern Broetz (DLR, Germany - bjoern.broetz@dlr.de)
  Lee de Mora (PML, UK - ledm@pml.ac.uk)
  Niels Drost (NLESC, Netherlands - n.drost@esciencecenter.nl)
  Nikolay Koldunov (AWI, Germany - nikolay.koldunov@awi.de)
  Axel Lauer (DLR, Germany - axel.lauer@dlr.de)
  Benjamin Mueller (LMU, Germany - b.mueller@iggf.geo.uni-muenchen.de)
  Valeriu Predoi (URead, UK - valeriu.predoi@ncas.ac.uk)
  Mattia Righi (DLR, Germany - mattia.righi@dlr.de)
  Manuel Schlund (DLR, Germany - manuel.schlund@dlr.de)
  Javier Vegas-Regidor (BSC, Spain - javier.vegas@bsc.es)
  Klaus Zimmermann (SMHI, Sweden - klaus.zimmermann@smhi.se)

For further help, please read the documentation at
http://esmvaltool.readthedocs.io. Have fun!
"""

# ESMValTool main script
#
# Authors:
# Bouwe Andela (NLESC, Netherlands - b.andela@esciencecenter.nl)
# Valeriu Predoi (URead, UK - valeriu.predoi@ncas.ac.uk)
# Mattia Righi (DLR, Germany - mattia.righi@dlr.de)

import datetime
import errno
import logging
import os
import shutil
import sys
from multiprocessing import cpu_count
from pkg_resources import iter_entry_points
import fire

from . import __version__
from ._config import configure_logging, read_config_user_file, DIAGNOSTICS_PATH
from ._recipe import TASKSEP, read_recipe_file
from ._task import resource_usage_logger
from .cmor.check import CheckLevels

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


def process_recipe(recipe_file, config_user):
    """Process recipe."""
    if not os.path.isfile(recipe_file):
        raise OSError(errno.ENOENT, "Specified recipe file does not exist",
                      recipe_file)

    timestamp1 = datetime.datetime.utcnow()
    timestamp_format = "%Y-%m-%d %H:%M:%S"

    logger.info(
        "Starting the Earth System Model Evaluation Tool v%s at time: %s UTC",
        __version__, timestamp1.strftime(timestamp_format))

    logger.info(70 * "-")
    logger.info("RECIPE   = %s", recipe_file)
    logger.info("RUNDIR     = %s", config_user['run_dir'])
    logger.info("WORKDIR    = %s", config_user["work_dir"])
    logger.info("PREPROCDIR = %s", config_user["preproc_dir"])
    logger.info("PLOTDIR    = %s", config_user["plot_dir"])
    logger.info(70 * "-")

    n_processes = config_user['max_parallel_tasks'] or cpu_count()
    logger.info("Running tasks using at most %s processes", n_processes)

    logger.info(
        "If your system hangs during execution, it may not have enough "
        "memory for keeping this number of tasks in memory.")
    logger.info(
        "If you experience memory problems, try reducing "
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
        "Ending the Earth System Model Evaluation Tool v%s at time: %s UTC",
        __version__, timestamp2.strftime(timestamp_format))
    logger.info("Time for running the recipe was: %s", timestamp2 - timestamp1)


class Config():
    """Manage configuration files."""

    @staticmethod
    def get_config_user(overwrite=False, target_path=None):
        """
        Copy default config-user.yml file to a given path.

        Parameters
        ----------
        overwrite: boolean
            Name of the recipe to get
        target_path: str
            If not provided, the file will be copied to
            .esmvaltool in the user's home.

        """
        if not target_path:
            target_path = os.path.expanduser('~/.esmvaltool/config-user.yml')
        if not overwrite and os.path.isfile(target_path):
            logger.info('Copy aborted. File %s already exists.')
        shutil.copy2(
            os.path.join(os.path.dirname(__file__), 'config-user.yml'),
            target_path)

    @staticmethod
    def get_config_developer(overwrite=False, target_path=None):
        """
        Copy default config-developer file to a given path.

        Parameters
        ----------
        overwrite: boolean
            Name of the recipe to get
        target_path: str
            If not provided, the file will be copied to
            .esmvaltool in the user's home.

        """
        if not target_path:
            target_path = os.path.expanduser(
                '~/.esmvaltool/config-developer.yml')
        if not overwrite and os.path.isfile(target_path):
            logger.info('Copy aborted. File %s already exists.')
        shutil.copy2(
            os.path.join(os.path.dirname(__file__), 'config-developer.yml'),
            target_path)


class Recipes():
    """Utilities to manage recipes."""

    @staticmethod
    def list():
        """List installed recipes."""
        configure_logging(output=None, console_log_level='info')
        recipes_folder = os.path.join(DIAGNOSTICS_PATH, 'recipes')
        logger.info('Installed recipes:')
        logger.info('==================')
        for root, _, files in os.walk(recipes_folder):
            root = os.path.relpath(root, recipes_folder)
            if root == '.':
                root = ''
            if root:
                logger.info('')
                logger.info(root.upper())
                logger.info('-' * len(root))
            for filename in files:
                if filename.endswith('.yml'):
                    logger.info(os.path.join(root, filename))

    @staticmethod
    def get(recipe):
        """
        Get a copy of any installed recipe in the current path.

        Parameters
        ----------
        recipe: str
            Name of the recipe to get
        """
        configure_logging(output=None, console_log_level='info')
        installed_recipe = os.path.join(DIAGNOSTICS_PATH, 'recipes', recipe)
        if not os.path.exists(installed_recipe):
            ValueError(
                f'Recipe {recipe} not found. To list all available recipes, '
                'execute "esmvaltool list"')
        logger.info('Copying installed recipe to the current folder...')
        shutil.copy(installed_recipe, recipe)
        logger.info('Recipe %s successfully copied', recipe)


class ESMValTool():
    """ESMValTool main executable."""

    def __init__(self):
        self.recipes = Recipes()
        self.config = Config()
        self._extra_packages = {}
        for entry_point in iter_entry_points('esmvaltool_commands'):
            self._extra_packages[entry_point.dist.project_name] = \
                entry_point.dist.version
            if hasattr(self, entry_point.name):
                logger.error(
                    'Registered command %s already exists', entry_point.name)
                continue
            self.__setattr__(entry_point.name, entry_point.load()())

    def version(self):
        """Show versions of ESMValTool packages."""
        print(f'ESMValCore: {__version__}')
        for project, version in self._extra_packages.items():
            print(f'{project}: {version}')

    @staticmethod
    def run(recipe, config_file=None, max_datasets=None, max_years=None,
            skip_nonexistent=False, synda_download=False, diagnostics=None,
            check_level='default', **kwargs):
        """
        Execute an ESMValTool recipe.

        Parameters
        ----------
        recipe : str
            Recipe to run, as either the name of an installed recipe or the
            path to a non-installed one
        config_file: str, optional
            Config file to use. If not provided will load 
            ${HOME}/.esmvaltool/config.user.yml if it exists
        max_datasets: int, optional
            Maximum number of datasets to compute
        max_years: int, optional
            Maximum number of years to compute
        skip_nonexistent: bool, optional
            If True, do not fail if data for some datasets is missing
        synda_download: bool, optional
            If True, download missing data using Synda if possible
        diagnostics: list(str), optional
            Only run the named diagnostics from the recipe
        check_level: str, optional
            Configure the severity of the errors that will make the CMOR check
            fail. Possible values:
                - ignore: all errors will be reported as warnings
                - relaxed: only fail if there are critical errors
                - default: fail if there are any errors
                -strict: fail if there are any warnings
        """
        if not os.path.exists(recipe):
            installed_recipe = os.path.join(
                DIAGNOSTICS_PATH, 'recipes', recipe)
            if os.path.exists(installed_recipe):
                recipe = installed_recipe
        recipe = os.path.abspath(
            os.path.expandvars(os.path.expanduser(recipe)))

        recipe_name = os.path.splitext(os.path.basename(recipe))[0]

        cfg = read_config_user_file(config_file, recipe_name, kwargs)

        # Create run dir
        if os.path.exists(cfg['run_dir']):
            print("ERROR: run_dir {} already exists, aborting to "
                  "prevent data loss".format(cfg['output_dir']))
        os.makedirs(cfg['run_dir'])

        # configure logging
        log_files = configure_logging(
            output=cfg['run_dir'], console_log_level=cfg['log_level'])

        # log header
        logger.info(HEADER)

        logger.info("Using config file %s", config_file)
        logger.info("Writing program log files to:\n%s", "\n".join(log_files))

        cfg['skip-nonexistent'] = skip_nonexistent
        cfg['diagnostics'] = {
            pattern if TASKSEP in pattern else pattern + TASKSEP + '*'
            for pattern in diagnostics or ()
        }
        cfg['check_level'] = CheckLevels[check_level.upper()]
        cfg['synda_download'] = synda_download

        def _check_limit(limit, value):
            if value is not None and value < 1:
                raise ValueError("--{} should be larger than 0.".format(
                    limit.replace('_', '-')))
            cfg[limit] = value

        _check_limit('max_datasets', max_datasets)
        _check_limit('max_years', max_years)

        resource_log = os.path.join(cfg['run_dir'], 'resource_usage.txt')
        with resource_usage_logger(pid=os.getpid(), filename=resource_log):
            process_recipe(recipe_file=recipe, config_user=cfg)

        if os.path.exists(cfg["preproc_dir"]) and cfg["remove_preproc_dir"]:
            logger.info("Removing preproc containing preprocessed data")
            logger.info("If this data is further needed, then")
            logger.info("set remove_preproc_dir to false in config")
            shutil.rmtree(cfg["preproc_dir"])
        logger.info("Run was successful")
        return cfg


def run():
    """Run the `esmvaltool` program, logging any exceptions."""
    try:
        fire.Fire(ESMValTool())
    except fire.core.FireExit as ex:
        sys.exit(ex.code)
    except Exception:  # noqa
        if not logger.handlers:
            # Add a logging handler if main failed to do so.
            logging.basicConfig()
        logger.exception(
            "Program terminated abnormally, see stack trace "
            "below for more information",
            exc_info=True)
        logger.info(
            "If you suspect this is a bug or need help, please open an issue "
            "on https://github.com/ESMValGroup/ESMValTool/issues and attach "
            "the run/recipe_*.yml and run/main_log_debug.txt files from the "
            "output directory.")
        sys.exit(1)
