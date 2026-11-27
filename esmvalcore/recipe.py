"""Work with recipes.

Validation
----------
Recipes are checked against a `JSON Schema <https://json-schema.org>`__ that
describes their structure and the arguments of all preprocessor functions.
The same schema can be used by editors to check recipes while writing them,
refer to :ref:`recipe_validation` for details.

For example, to find the problems in a recipe, with their line numbers:

>>> from esmvalcore.recipe import validate_recipe
>>> text = '''
... documentation:
...   title: Example
...   description: Example recipe.
...   authors: [andela_bouwe]
... diagnostics:
...   example:
...     scripts: null
... preprocessors:
...   pp:
...     regird: {}
... '''
>>> for issue in validate_recipe(text):
...     print(issue)
line 11: preprocessors > pp > regird: unknown preprocessor function. Did you mean 'regrid'?

Formatting
----------
Recipes can be written in a standard format with :func:`format_recipe`.

For example, to write a recipe in the standard format, with the facets of the
datasets in the standard order:

>>> from esmvalcore.recipe import format_recipe
>>> text = '''
... datasets:
...   - &cesm {project: CMIP6, dataset: CESM2, exp: historical}
... diagnostics:
...   example:
...     additional_datasets: [*cesm]
...     scripts: null
... '''
>>> print(format_recipe(text), end="")
datasets:
  - &cesm {dataset: CESM2, exp: historical, project: CMIP6}
<BLANKLINE>
diagnostics:
  example:
    additional_datasets:
      - *cesm
    scripts: null
"""

from esmvalcore._recipe.json_schema import generate_recipe_schema
from esmvalcore._recipe.validate import RecipeIssue, validate_recipe
from esmvalcore._recipe.writer import format_recipe

__all__ = [
    "RecipeIssue",
    "format_recipe",
    "generate_recipe_schema",
    "validate_recipe",
]
