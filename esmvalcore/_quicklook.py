"""
Creates recipes for EMAC simulation.

"""

import datetime
import glob
import os
import re
from pathlib import Path, PurePath
import yaml

import logging
from jinja2 import Template

from ._config import read_config_user_file, read_config_developer_file, get_project_config

logger = logging.getLogger(__name__)


def create_recipes(cfg, start, stop):
    quicklook_recipe_dir = cfg['quicklook_recipe_dir']

    try:
        Path(quicklook_recipe_dir)
    except TypeError:
        quicklook_recipe_dir = PurePath(
            Path(__file__)).parent.joinpath('quicklook')

    recipes = cfg['quicklook_recipes']
    run_id = cfg['quicklook_run_id']
    output_dir = os.path.join(cfg['quicklook_output_dir'], run_id)

    logger.debug("Creating directory %s", output_dir)
    os.makedirs(output_dir, exist_ok=True)
    for recipe in recipes:
        # TODO: Remove hardcoded filename elements from here and from the filenames
        recipe_name = 'template_recipe_emac_{0}.yml'.format(recipe)
        with open(os.path.join(quicklook_recipe_dir, recipe_name)) as datei:
            recipe_template = datei.read()
            template = Template(recipe_template)
            content = template.render(
                dataset="{" +
                "dataset: EMAC, project: EMAC, exp:{0}, start_year: {1}, end_year: {2}"
                .format(run_id, start, stop)) + "}"

        with open(os.path.join(output_dir, recipe_name), 'w') as f:
            logger.debug("Writing %s to  %s", recipe_name, output_dir)
            f.write(content)


def run():
    # TODO: remove hard coding
    cfg = read_config_user_file('/home/bjoern/dev/config/config-user_BB.yml',
                                'RECIPE')
    logging.basicConfig(filename=os.path.join(cfg['quicklook_output_dir'],
                                              'logfile.log'),
                        level=logging.INFO,
                        format='%(asctime)s %(message)s')
    create_recipes(cfg, 1900, 1903)
    logging.info("Simulation {0} computed for {1}-{2}".format(
        cfg['dataset'], 1900, 1903))


if __name__ == "__main__":
    run()
