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
http://docs.esmvaltool.org. Have fun!
"""

import logging
from pathlib import Path

import fire
from pkg_resources import iter_entry_points

from esmvalcore import config, session

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


def process_recipe(recipe_file):
    """Process recipe."""
    import datetime
    import shutil

    from . import __version__
    from ._recipe import read_recipe_file
    if not recipe_file.exists():
        import errno
        raise OSError(errno.ENOENT, "Specified recipe file does not exist",
                      recipe_file)

    timestamp1 = datetime.datetime.utcnow()
    timestamp_format = "%Y-%m-%d %H:%M:%S"

    logger.info(
        "Starting the Earth System Model Evaluation Tool v%s at time: %s UTC",
        __version__, timestamp1.strftime(timestamp_format))

    logger.info(70 * "-")
    logger.info("RECIPE   = %s", recipe_file)
    logger.info("RUNDIR     = %s", session.run_dir)
    logger.info("WORKDIR    = %s", session.work_dir)
    logger.info("PREPROCDIR = %s", session.preproc_dir)
    logger.info("PLOTDIR    = %s", session.plot_dir)
    logger.info(70 * "-")

    from multiprocessing import cpu_count
    n_processes = config['max_parallel_tasks'] or cpu_count()
    logger.info("Running tasks using at most %s processes", n_processes)

    logger.info(
        "If your system hangs during execution, it may not have enough "
        "memory for keeping this number of tasks in memory.")
    logger.info("If you experience memory problems, try reducing "
                "'max_parallel_tasks' in your user configuration file.")

    if config['compress_netcdf']:
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
    recipe = read_recipe_file(recipe_file)
    logger.debug("Recipe summary:\n%s", recipe)

    # run
    recipe.run()

    # End time timing
    timestamp2 = datetime.datetime.utcnow()
    logger.info(
        "Ending the Earth System Model Evaluation Tool v%s at time: %s UTC",
        __version__, timestamp2.strftime(timestamp_format))
    logger.info("Time for running the recipe was: %s", timestamp2 - timestamp1)


class ConfigUtils():
    """Manage ESMValTool's configuration.

    This group contains utilities to manage ESMValTool configuration
    files.
    """
    @classmethod
    def get(cls, name, overwrite=False, path=None):
        """Initialize default config-user.yml to a given path.

        Copy default config-user.yml file to a given path or, if a path is
        not provided, install it in the default `${HOME}/.esmvaltool` folder.

        Parameters
        ----------
        name: str
            Name of the config to get, i.e. jasmin/dkrz/ethz
        overwrite: boolean
            Overwrite an existing file.
        path: str
            If not provided, the file will be written to
            .esmvaltool in the user's home.
        """
        from ._config import configure_logging
        configure_logging(console_log_level='info')

        blank_lines = ['\n']

        config_file = Path(__file__).with_name('config-user.yml')
        config_lines = open(config_file, 'r').readlines()

        drs_file = Path(__file__).with_name(f'drs-{name}.yml')
        drs_lines = open(drs_file, 'r').readlines()
        drs_lines = [line for line in drs_lines if not line.startswith('---')]
        config_lines = config_lines + blank_lines + drs_lines

        filename = f'config-{name}.yml'

        if not path:
            path = Path('~/.esmvaltool').expanduser() / filename
        else:
            path = Path(path)

        if path.exists():
            if overwrite:
                logger.info('Overwriting file %s.', path)
            else:
                logger.info('Get config aborted. File %s already exists.',
                            path)
                return

        with open(path, 'w') as f:
            f.writelines(config_lines)

        logger.info('Writing config file to %s.', path)

    @classmethod
    def list(cls):

        pass

        # list user configs in ~/.esmvalcore/
        # list available configs


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

        from ._config import DIAGNOSTICS_PATH, configure_logging
        configure_logging(console_log_level='info')
        recipes_folder = os.path.join(DIAGNOSTICS_PATH, 'recipes')
        logger.info("Showing recipes installed in %s", recipes_folder)
        print('# Installed recipes')
        for root, _, files in os.walk(recipes_folder):
            root = os.path.relpath(root, recipes_folder)
            if root == '.':
                root = ''
            if root:
                print(f"\n# {root.replace(os.sep, ' - ').title()}")
            for filename in files:
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
        import os
        import shutil

        from ._config import DIAGNOSTICS_PATH, configure_logging
        configure_logging(console_log_level='info')
        installed_recipe = os.path.join(DIAGNOSTICS_PATH, 'recipes', recipe)
        if not os.path.exists(installed_recipe):
            ValueError(
                f'Recipe {recipe} not found. To list all available recipes, '
                'execute "esmvaltool list"')
        logger.info('Copying installed recipe to the current folder...')
        shutil.copy(installed_recipe, os.path.basename(recipe))
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
        import os

        from ._config import DIAGNOSTICS_PATH, configure_logging
        configure_logging(console_log_level='info')
        installed_recipe = os.path.join(DIAGNOSTICS_PATH, 'recipes', recipe)
        if not os.path.exists(installed_recipe):
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
        self.config = ConfigUtils()
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

    @staticmethod
    def run(recipe,
            config_file=None,
            max_datasets=None,
            max_years=None,
            skip_nonexistent=False,
            synda_download=False,
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
        max_datasets: int, optional
            Maximum number of datasets to use.
        max_years: int, optional
            Maximum number of years to use.
        skip_nonexistent: bool, optional
            If True, the run will not fail if some datasets are not available.
        synda_download: bool, optional
            If True, the tool will try to download missing data using Synda.
        diagnostics: list(str), optional
            Only run the selected diagnostics from the recipe.
        check_level: str, optional
            Configure the sensitivity of the CMOR check. Possible values are:
            `ignore` (all errors will be reported as warnings),
            `relaxed` (only fail if there are critical errors),
            default (fail if there are any errors),
            strict (fail if there are any warnings).
        """
        import os
        import shutil

        from esmvalcore import config, session

        from ._config import DIAGNOSTICS_PATH, configure_logging

        if config_file:
            from esmvalcore._config_object import _load_user_config
            _load_user_config(config_file)

        recipe = Path(recipe)

        if not recipe.exists():
            installed_recipe = Path(DIAGNOSTICS_PATH, 'recipes', recipe)
            if installed_recipe.exists():
                recipe = installed_recipe

        recipe = recipe.expanduser().absolute()
        recipe_name = recipe.stem

        # init and create run dir
        session.init_session_dir(recipe_name)
        session.run_dir.mkdir(parents=True)

        # configure logging
        log_files = configure_logging(output_dir=str(session.run_dir),
                                      console_log_level=config['log_level'])

        # log header
        logger.info(HEADER)

        if config_file:
            logger.info("Using config file %s", config_file)
        logger.info("Writing program log files to:\n%s", "\n".join(log_files))

        # Update config with CLI options
        config['skip-nonexistent'] = skip_nonexistent
        config['diagnostics'] = diagnostics
        config['check_level'] = check_level
        config['synda_download'] = synda_download
        config['max_datasets'] = max_datasets
        config['max_years'] = max_years

        # Add additional command line arguments to config
        config.update(kwargs)

        resource_log = session.run_dir / 'resource_usage.txt'

        from ._task import resource_usage_logger
        with resource_usage_logger(pid=os.getpid(), filename=resource_log):
            process_recipe(recipe_file=recipe)

        if session.preproc_dir.exists() and config["remove_preproc_dir"]:
            logger.info("Removing preproc containing preprocessed data")
            logger.info("If this data is further needed, then")
            logger.info("set remove_preproc_dir to false in config-user.yml")
            shutil.rmtree(session.preproc_dir)
        logger.info("Run was successful")


def run():
    """Run the `esmvaltool` program, logging any exceptions."""
    import sys

    # Workaroud to avoid using more for the output

    def display(lines, out):
        text = "\n".join(lines) + "\n"
        out.write(text)

    fire.core.Display = display

    try:
        fire.Fire(ESMValTool())
    except fire.core.FireExit:
        raise
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
