"""Creates recipes for EMAC simulation."""

import logging
import os

import yaml

logger = logging.getLogger(__name__)


def create_recipe(cfg):
    """Create recipe for quicklook mode and return its path."""
    print("Creating quicklook recipe")
    start = cfg['quicklook'].get('start')
    end = cfg['quicklook'].get('end')
    # TODO: We should rename "recipes" to "diagnostics" in this context
    recipes = cfg['quicklook'].get('recipes')
    run_id = cfg['quicklook'].get('dataset-id')
    preproc_dir = os.path.join(cfg['quicklook']['preproc_dir'], run_id)
    recipe_dir = cfg['quicklook'].get('recipe_dir')

    logger.debug("Creating directory %s", preproc_dir)
    os.makedirs(preproc_dir, exist_ok=True)
    all_diagnostics = {}
    for recipe in recipes:
        recipe_name = f'diagnostics_{recipe}.yml'
        with open(os.path.join(recipe_dir, recipe_name)) as stream:
            diagnostics = yaml.load(stream, Loader=yaml.FullLoader)
            for (diag_name, diag_content) in diagnostics.items():
                diag_name = diag_name.format(run_id=run_id)
                all_diagnostics[diag_name] = diag_content
    with open(os.path.join(recipe_dir, 'general.yml')) as stream:
        out = yaml.load(stream, Loader=yaml.FullLoader)

    out['diagnostics'] = all_diagnostics
    out['datasets'] = [{
        'dataset': run_id,
        'project': 'EMAC',
        'start_year': start,
        'end_year': end
    }]
    path_to_recipe = os.path.join(preproc_dir, 'recipe_quicklook.yml')
    with open(path_to_recipe, 'w') as stream:
        logger.debug("Writing %s to %s", recipe_name, preproc_dir)
        stream.write(yaml.dump(out))
    return path_to_recipe
