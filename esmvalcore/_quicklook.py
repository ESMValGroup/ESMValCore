"""Creates recipes for EMAC simulation."""

import os

import yaml


def _get_multi_plot_diags(plot_scripts):
    """Get diagnostic to plot multiple datasets in one plot."""
    for script_body in plot_scripts.values():
        script_body['multi_dataset_plot'] = True
        script_body['read_all_available_datasets'] = True
    diag = {
        'description': 'Plot multiple runs in one plot',
        'scripts': plot_scripts,
    }
    return {'multi_run_plots': diag}


def _get_single_plot_diags(recipe_dir, recipes, plot_scripts, run_id):
    """Get desired single plot diagnostics."""
    all_diagnostics = {}
    for recipe in recipes:
        recipe_name = f'diagnostics_{recipe}.yml'
        with open(os.path.join(recipe_dir, recipe_name)) as stream:
            diagnostics = yaml.load(stream, Loader=yaml.FullLoader)
        for (diag_name, diag_content) in diagnostics.items():
            diag_name = diag_name.format(run_id=run_id)
            diag_content['scripts'] = plot_scripts
            all_diagnostics[diag_name] = diag_content
    return all_diagnostics


def create_recipe(cfg):
    """Create recipe for quicklook mode and return its path."""
    print("INFO: Creating quicklook recipe")
    opts = cfg['quicklook']
    # TODO: We should rename "recipes" to "diagnostics" in this context
    recipes = opts.get('recipes', [])
    run_id = opts['dataset-id']
    preproc_dir = os.path.join(opts['preproc_dir'], run_id)
    recipe_dir = opts['recipe_dir']
    if not os.path.isdir(preproc_dir):
        os.makedirs(preproc_dir)
        print(f"INFO: Created non-existent recipe directory '{preproc_dir}'")

    # Get Header
    with open(os.path.join(recipe_dir, 'general.yml')) as stream:
        out = yaml.load(stream, Loader=yaml.FullLoader)

    # Get plot scripts
    with open(os.path.join(recipe_dir, 'plot_scripts.yml')) as stream:
        plot_scripts = yaml.load(stream, Loader=yaml.FullLoader)

    # Get diagnostics and datasets
    if opts['multi_run_plots']:
        out['diagnostics'] = _get_multi_plot_diags(plot_scripts)
    else:
        out['diagnostics'] = _get_single_plot_diags(recipe_dir, recipes,
                                                    plot_scripts, run_id)
        out['datasets'] = [{
            'dataset': run_id,
            'project': 'EMAC',
            'start_year': opts['start'],
            'end_year': opts['end'],
        }]

    # Write recipe
    path_to_recipe = os.path.join(preproc_dir, 'recipe_quicklook.yml')
    with open(path_to_recipe, 'w') as stream:
        stream.write(yaml.dump(out))
    print(f"INFO: Wrote quicklook recipe '{path_to_recipe}'")
    return path_to_recipe
