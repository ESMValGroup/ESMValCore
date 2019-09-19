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
import shutil
import sys
from copy import deepcopy
from multiprocessing import cpu_count

from . import __version__
from ._config import DIAGNOSTICS_PATH, configure_logging, read_config_user_file
from ._recipe import TASKSEP, read_recipe_file
from ._recipe_checks import RecipeError
from ._task import resource_usage_logger

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
        'recipes',
        help='Path or name of the yaml recipe file(s)',
        nargs='+',
    )
    parser.add_argument(
        '-v',
        '--version',
        action='version',
        version=__version__,
        help="return ESMValTool's version number and exit",
    )
    parser.add_argument(
        '-c',
        '--config-file',
        default=os.path.join(os.path.dirname(__file__), 'config-user.yml'),
        help='Config file',
    )
    parser.add_argument(
        '-s',
        '--synda-download',
        action='store_true',
        help='Download input data using synda. This requires a working '
        'synda installation.')
    parser.add_argument('--exit-on-error',
                        action='store_true',
                        help='Exit immediately when a recipe fails to run.')
    parser.add_argument(
        '--max-datasets',
        type=int,
        help='Try to limit the number of datasets used to MAX_DATASETS.',
    )
    parser.add_argument(
        '--max-years',
        type=int,
        help='Limit the number of years to MAX_YEARS.',
    )
    parser.add_argument(
        '--skip-nonexistent',
        action='store_true',
        help="Skip datasets that cannot be found.",
    )
    parser.add_argument(
        '--diagnostics',
        nargs='*',
        help="Only run the named diagnostics from the recipe.",
    )
    args = parser.parse_args()
    return args


def configure(args):
    """Define the `esmvaltool` program."""
    recipes = []
    for recipe in args.recipes:
        if not os.path.exists(recipe):
            installed_recipe = os.path.join(DIAGNOSTICS_PATH, 'recipes',
                                            recipe)
            if os.path.exists(installed_recipe):
                recipe = installed_recipe
        recipe = os.path.abspath(os.path.expandvars(
            os.path.expanduser(recipe)))
        recipes.append(recipe)

    config_file = os.path.abspath(
        os.path.expandvars(os.path.expanduser(args.config_file)))

    # Read user config file
    if not os.path.exists(config_file):
        print("ERROR: config file {} does not exist".format(config_file))

    cfg = read_config_user_file(config_file)

    cfg['skip-nonexistent'] = args.skip_nonexistent
    cfg['diagnostics'] = {
        pattern if TASKSEP in pattern else pattern + TASKSEP + '*'
        for pattern in args.diagnostics or ()
    }
    cfg['synda_download'] = args.synda_download
    cfg['exit_on_error'] = args.exit_on_error
    for limit in ('max_datasets', 'max_years'):
        value = getattr(args, limit)
        if value is not None:
            if value < 1:
                raise ValueError("--{} should be larger than 0.".format(
                    limit.replace('_', '-')))
            cfg[limit] = value

    return cfg, recipes


def configure_recipe_paths(cfg, recipe):
    """Create a copy of cfg with recipe specific paths."""
    cfg = deepcopy(cfg)

    # insert a directory date_time_recipe_usertag in the output paths
    recipe_name = os.path.splitext(os.path.basename(recipe))[0]
    now = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    subdir = '_'.join((recipe_name, now))
    cfg['output_dir'] = os.path.join(cfg['output_dir'], subdir)
    if 'tmp_dir' in cfg:
        cfg['tmp_dir'] = os.path.join(cfg['tmp_dir'], subdir)

    # Define directories
    if 'tmp_dir' in cfg and cfg['remove_preproc_dir']:
        cfg['preproc_dir'] = os.path.join(cfg['tmp_dir'], 'preproc')
    else:
        cfg['preproc_dir'] = os.path.join(cfg['output_dir'], 'preproc')
    cfg['run_dir'] = os.path.join(cfg['output_dir'], 'run')
    cfg['work_dir'] = os.path.join(cfg['output_dir'], 'work')
    cfg['plot_dir'] = os.path.join(cfg['output_dir'], 'plots')

    return cfg


def _run_recipe(recipe_file, cfg):
    """Run the recipe."""
    # Create output dir
    output_dir = cfg['output_dir']
    try:
        os.makedirs(output_dir)
    except OSError:
        print(f"ERROR: output_dir {output_dir} already exists, aborting to "
              "prevent data loss")
        raise

    if 'tmp_dir' in cfg:
        tmp_dir = cfg['tmp_dir']
        try:
            os.makedirs(tmp_dir)
        except OSError:
            print(f"ERROR: tmp_dir {tmp_dir} already exists, aborting to "
                  "prevent data loss")
            raise

    # configure logging
    log_files = configure_logging(output=cfg['output_dir'],
                                  console_log_level=cfg['log_level'])

    # log header
    logger.info(HEADER)
    logger.info("Starting the Earth System Model Evaluation Tool v%s",
                __version__)
    logger.info("Writing log files to:\n%s", "\n".join(log_files))

    resource_log = os.path.join(cfg['output_dir'], 'resource_usage.txt')
    with resource_usage_logger(pid=os.getpid(), filename=resource_log):
        process_recipe(recipe_file, cfg)


def process_recipe(recipe_file, cfg):
    """Process recipe."""
    if not os.path.isfile(recipe_file):
        raise OSError(errno.ENOENT, "Specified recipe file does not exist",
                      recipe_file)

    start = datetime.datetime.utcnow()
    logger.info("Started running recipe %s at %s UTC", recipe_file, start)

    logger.info(70 * "-")
    logger.info("Recipe file = %s", recipe_file)
    logger.info("output_dir  = %s", cfg["output_dir"])
    logger.info("work_dir    = %s", cfg["work_dir"])
    logger.info("plot_dir    = %s", cfg["plot_dir"])
    logger.info("run_dir     = %s", cfg['run_dir'])
    logger.info("preproc_dir = %s", cfg["preproc_dir"])

    logger.info(70 * "-")

    logger.info("Running tasks using at most %s processes",
                cfg['max_parallel_tasks'] or cpu_count())

    logger.info(
        "If your system hangs during execution, it may not have enough "
        "memory for keeping this number of tasks in memory.")
    logger.info("In that case, try reducing 'max_parallel_tasks' in your user "
                "configuration file.")

    if cfg['compress_netcdf']:
        logger.warning(
            "You have enabled NetCDF compression. Accesing .nc files can be "
            "much slower than expected if your access pattern does not match "
            "their internal pattern. Make sure to specify the expected "
            "access pattern in the recipe as a parameter to the 'save' "
            "preprocessor function. If the problem persists, try disabling "
            "NetCDF compression.")

    # copy recipe to run_dir for future reference
    shutil.copy2(recipe_file, cfg['output_dir'])

    # parse recipe
    recipe = read_recipe_file(recipe_file, cfg)
    logger.debug("Recipe summary:\n%s", recipe)

    # run
    recipe.run()

    if cfg["remove_preproc_dir"]:
        logger.info("Removing preproc_dir %s containing preprocessed data",
                    cfg['preproc_dir'])
        logger.info("If this data is further needed, set remove_preproc_dir "
                    "to false in user configuration file.")
        shutil.rmtree(cfg["preproc_dir"])

    # End time timing
    logger.info("Successfully ran recipe %s in %s", recipe_file,
                datetime.datetime.utcnow() - start)


def run():
    """Run the `esmvaltool` program, logging any exceptions."""
    args = get_args()
    cfg, recipes = configure(args)
    errors = []

    for recipe_file in recipes:
        success = True
        recipe_cfg = configure_recipe_paths(cfg, recipe_file)
        try:
            _run_recipe(recipe_file, recipe_cfg)
        except RecipeError as exc:
            logger.error("An error occurred in recipe %s", recipe_file)
            logger.error("%s", exc)
            logger.debug("RecipeError:", exc_info=True)
            success = False
        except:  # noqa
            if not logger.handlers:
                # Add a logging handler if main failed to do so.
                logging.basicConfig()
            logger.exception(
                "Program terminated abnormally, see stack trace "
                "below for more information",
                exc_info=True)
            logger.info(
                "If you suspect this is a bug or need help, please open an "
                "issue on https://github.com/ESMValGroup/ESMValTool/issues "
                "and attach the files: %s and main_log_debug.txt from "
                "directory %s", os.path.basename(recipe_file),
                recipe_cfg['output_dir'])
            success = False
        finally:
            if 'tmp_dir' in recipe_cfg:
                logger.info("Removing tmp_dir %s", recipe_cfg['tmp_dir'])
                shutil.rmtree(recipe_cfg['tmp_dir'], ignore_errors=True)

        if success:
            logger.info("Successfully completed recipe %s", recipe_file)
        else:
            logger.error("Failed to run recipe %s", recipe_file)
            errors.append(recipe_file)
            if cfg['exit_on_error']:
                break

    if errors:
        logger.error("Error(s) occurred in recipe(s):\n%s", '\n'.join(errors))
        sys.exit(1)
    else:
        logger.info("Run was successful")
