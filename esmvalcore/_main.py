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

import argparse
import datetime
import errno
import logging
import os
import glob
import shutil
import sys
from multiprocessing import cpu_count

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


def get_args():
    """Define the `esmvaltool` command line."""
    # parse command line args
    parser = argparse.ArgumentParser(
        description=HEADER,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '-v',
        '--version',
        action='version',
        version=__version__,
        help="return ESMValTool's version number and exit")
    subparsers = parser.add_subparsers(required=True)

    run = subparsers.add_parser("run", help='Run recipe')
    run.set_defaults(func=main)
    run.add_argument('recipe', help='Path or name of the yaml recipe file')

    run.add_argument(
        '-c',
        '--config-file',
        default=os.path.join(os.path.dirname(__file__), 'config-user.yml'),
        help='Config file')
    run.add_argument(
        '-s',
        '--synda-download',
        action='store_true',
        help='Download input data using synda. This requires a working '
        'synda installation.')
    run.add_argument(
        '--max-datasets',
        type=int,
        help='Try to limit the number of datasets used to MAX_DATASETS.')
    run.add_argument(
        '--max-years',
        type=int,
        help='Limit the number of years to MAX_YEARS.')
    run.add_argument(
        '--skip-nonexistent',
        action='store_true',
        help="Skip datasets that cannot be found.")
    run.add_argument(
        '--diagnostics',
        nargs='*',
        help="Only run the named diagnostics from the recipe.")
    run.add_argument(
        '--check-level',
        type=str,
        choices=[
            val.name.lower() for val in CheckLevels if val != CheckLevels.DEBUG
        ],
        default='default',
        help="""
            Configure the severity of the errors that will make the CMOR check
            fail.
            Optional: true;
            Possible values:
            ignore: all errors will be reported as warnings
            relaxed: only fail if there are critical errors
            default: fail if there are any errors
            strict: fail if there are any warnings
        """
    )
    list_parser = subparsers.add_parser("list", help='List installed recipes')
    list_parser.set_defaults(func=list_recipes)

    get_parser = subparsers.add_parser("get", help='Get an installed recipe')
    get_parser.set_defaults(func=get_recipe)
    get_parser.add_argument('recipe', help='Name of the yaml recipe file')
    args = parser.parse_args()
    return args


def list_recipes(args):
    configure_logging(output=None, console_log_level='info')
    recipes_folder = os.path.join(DIAGNOSTICS_PATH, 'recipes')
    logger.info('Installed recipes:')
    logger.info('------------------')
    for path in glob.glob(os.path.join(recipes_folder, '*.yml')):
        logger.info(os.path.relpath(path, recipes_folder))


def get_recipe(args):
    configure_logging(output=None, console_log_level='info')
    installed_recipe = os.path.join(DIAGNOSTICS_PATH, 'recipes', args.recipe)
    if not os.path.exists(installed_recipe):
        ValueError(
            f'Recipe {args.recipe} not found. To list all available recipes, '
            'execute "esmvaltool list"')
    logger.info('Copying installed recipe to the current folder...')
    shutil.copy(installed_recipe, args.recipe)
    logger.info('Recipe %s successfully copied', args.recipe)


def main(args):
    """Define the `esmvaltool` program."""
    recipe = args.recipe
    if not os.path.exists(recipe):
        installed_recipe = os.path.join(
            DIAGNOSTICS_PATH, 'recipes', recipe)
        if os.path.exists(installed_recipe):
            recipe = installed_recipe
    recipe = os.path.abspath(os.path.expandvars(os.path.expanduser(recipe)))

    config_file = os.path.abspath(
        os.path.expandvars(os.path.expanduser(args.config_file)))

    # Read user config file
    if not os.path.exists(config_file):
        print("ERROR: config file {} does not exist".format(config_file))

    recipe_name = os.path.splitext(os.path.basename(recipe))[0]
    cfg = read_config_user_file(config_file, recipe_name)

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

    cfg['skip-nonexistent'] = args.skip_nonexistent
    cfg['diagnostics'] = {
        pattern if TASKSEP in pattern else pattern + TASKSEP + '*'
        for pattern in args.diagnostics or ()
    }
    cfg['check_level'] = CheckLevels[args.check_level.upper()]
    cfg['synda_download'] = args.synda_download
    for limit in ('max_datasets', 'max_years'):
        value = getattr(args, limit)
        if value is not None:
            if value < 1:
                raise ValueError("--{} should be larger than 0.".format(
                    limit.replace('_', '-')))
            cfg[limit] = value

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


def run():
    """Run the `esmvaltool` program, logging any exceptions."""
    args = get_args()
    try:
        conf = args.func(args)
    except:  # noqa
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
