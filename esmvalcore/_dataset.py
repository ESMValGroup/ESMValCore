import copy
import re

import yaml

from esmvalcore._data_finder import get_input_filelist
from esmvalcore._recipe_checks import RecipeError


class Dataset:

    def __init__(self, **facets):

        self.facets = facets

    def __eq__(self, other):
        return isinstance(other,
                          self.__class__) and self.facets == other.facets

    def __repr__(self):
        return repr(self.facets)

    def copy(self, **facets):
        # Is this a useful function? Maybe remove
        facets_ = copy.deepcopy(self.facets)
        facets_.update(facets)
        return self.__class__(**facets_)

    def find_files(self, config_user, debug=False):
        (input_files, dirnames, filenames) = get_input_filelist(
            variable=self.facets,
            rootpath=config_user['rootpath'],
            drs=config_user['drs'],
        )
        if debug:
            return (input_files, dirnames, filenames)
        return input_files


def _expand_tag(facets, input_tag):
    """Expand tags such as ensemble members or stardates to multiple datasets.

    Expansion only supports ensembles defined as strings, not lists.
    """
    expanded = []
    regex = re.compile(r'\(\d+:\d+\)')

    def expand_tag(variable, input_tag):
        tag = variable.get(input_tag, "")
        match = regex.search(tag)
        if match:
            start, end = match.group(0)[1:-1].split(':')
            for i in range(int(start), int(end) + 1):
                expand = copy.deepcopy(variable)
                expand[input_tag] = regex.sub(str(i), tag, 1)
                expand_tag(expand, input_tag)
        else:
            expanded.append(variable)

    tag = facets.get(input_tag, "")
    if isinstance(tag, (list, tuple)):
        for elem in tag:
            if regex.search(elem):
                raise RecipeError(
                    f"In variable {facets}: {input_tag} expansion "
                    f"cannot be combined with {input_tag} lists")
        expanded.append(facets)
    else:
        expand_tag(facets, input_tag)

    return expanded


def datasets_from_recipe(recipe):

    with open(recipe, 'r') as file:
        recipe = yaml.safe_load(file)

    datasets = []

    for diagnostic in recipe['diagnostics']:
        for variable_group in recipe['diagnostics'][diagnostic].get(
                'variables', {}):
            # Read variable from recipe
            recipe_variable = recipe['diagnostics'][diagnostic]['variables'][
                variable_group]
            if recipe_variable is None:
                recipe_variable = {}
            # Read datasets from recipe
            recipe_datasets = (recipe.get('datasets', []) +
                               recipe['diagnostics'][diagnostic].get(
                                   'additional_datasets', []) +
                               recipe_variable.get('additional_datasets', []))

            for recipe_dataset in recipe_datasets:
                facets = copy.deepcopy(recipe_variable)
                facets.pop('additional_datasets', None)
                facets.pop('preprocessor', None)
                facets.update(copy.deepcopy(recipe_dataset))
                facets['diagnostic'] = diagnostic
                facets['variable_group'] = variable_group
                if 'short_name' not in facets:
                    facets['short_name'] = variable_group

                for facets0 in _expand_tag(facets, 'ensemble'):
                    for facets1 in _expand_tag(facets0, 'sub_experiment'):
                        dataset = Dataset(**facets1)
                        datasets.append(dataset)
    return datasets


def datasets_to_recipe(datasets):
    pass
