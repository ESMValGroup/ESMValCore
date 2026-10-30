"""Validate recipes.

Recipes are checked against a `JSON Schema <https://json-schema.org>`__ that
describes their structure and the arguments of all preprocessor functions.
The same schema can be used by editors to check recipes while writing them,
see :ref:`recipe_editor_support`.

Example
-------
Find the problems in a recipe, with their line numbers:

>>> import yaml
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
>>> for issue in validate_recipe(yaml.safe_load(text), text):
...     print(issue)
line 11: preprocessors > pp > regird: unknown preprocessor function. Did you mean 'regrid'?
"""

from esmvalcore._recipe.json_schema import (
    RecipeIssue,
    generate_recipe_schema,
    validate_recipe,
    write_recipe_schema,
)

__all__ = [
    "RecipeIssue",
    "generate_recipe_schema",
    "validate_recipe",
    "write_recipe_schema",
]
