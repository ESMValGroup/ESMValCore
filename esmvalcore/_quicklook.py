"""
Creates recipes for EMAC simulation.

"""

import datetime
import glob
import logging
import os
import re
from pathlib import Path, PurePath

import yaml
from jinja2 import Template

from ._config import (get_project_config, read_config_developer_file,
                      read_config_user_file)

logger = logging.getLogger(__name__)

DEFAULT_TEMPLATES = Path(__file__).parent.joinpath('quicklook').resolve()


def create_recipe(cfg):
    start = cfg['quicklook'].get('start')
    stop = cfg['quicklook'].get('stop')
    recipes = cfg['quicklook'].get('recipes')
    run_id = cfg['quicklook'].get('dataset-id')
    output_dir = os.path.join(cfg['quicklook'].get('output_dir'), run_id)
    recipe_dir = cfg['quicklook'].get('recipe_dir', DEFAULT_TEMPLATES)

    logger.debug("Creating directory %s", output_dir)
    os.makedirs(output_dir, exist_ok=True)
    for recipe in recipes:
        # TODO: Remove hardcoded filename elements from here and from the
        # filenames
        recipe_name = 'template_recipe_emac_{0}.yml'.format(recipe)
        with open(os.path.join(recipe_dir, recipe_name)) as datei:
            recipe_template = datei.read()
            template = Template(recipe_template)
            content = template.render(
                dataset="{dataset: EMAC," + "project: EMAC," +
                " exp:{0}, start_year: {1}, end_year: {2}".format(
                    run_id, start, stop)) + "}"
        path_to_recipe = os.path.join(output_dir, recipe_name)
        with open(path_to_recipe, 'w') as f:
            logger.debug("Writing %s to  %s", recipe_name, output_dir)
            f.write(content)
    return path_to_recipe
