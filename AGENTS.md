# AGENTS.md

Guidance for AI coding agents working in the ESMValCore repository.
The full contributor guide is in [doc/contributing.rst](doc/contributing.rst);
this file summarizes what matters most when making changes.

## Project overview

ESMValCore is the core of [ESMValTool](https://github.com/ESMValGroup/ESMValTool):
it finds, fixes, and preprocesses Earth system model data (CMIP, CORDEX,
observations, native model output) according to a YAML *recipe* and then runs
diagnostic scripts. It is built on [iris](https://scitools-iris.readthedocs.io)
cubes and [Dask](https://docs.dask.org) for lazy, out-of-core computation.

Key locations:

- `esmvalcore/_main.py` – the `esmvaltool` command line interface.
- `esmvalcore/_recipe/` – recipe parsing, validation (`_recipe/check.py`), and
  conversion to datasets and tasks.
- `esmvalcore/dataset.py` – the public `Dataset` API.
- `esmvalcore/io/` – data sources (e.g. local files, ESGF).
- `esmvalcore/cmor/` – CMOR tables, checks, and dataset-specific fixes
  (`cmor/_fixes/<project>/<dataset>.py`).
- `esmvalcore/preprocessor/` – preprocessor functions; derived variables live
  in `preprocessor/_derive/`.
- `esmvalcore/config/` – configuration handling, including extra facets.
- `doc/` – Sphinx documentation (reStructuredText).
- `tests/unit/`, `tests/integration/`, `tests/sample_data/` – tests.

## Development environment

The development environment is managed with [pixi](https://pixi.sh):

```bash
pixi run test                       # run the full test suite
pixi run doc                        # build the documentation into html/
pixi run pytest tests/unit/test_dataset.py   # run a single test file
pixi run pre-commit run --all-files # run all linters and formatters
```

Useful pytest options:

- `-m "not online"` skips tests that need internet access.
- `-m "not use_sample_data"` skips the slow sample-data tests.
- `-n auto` runs tests in parallel (pytest-xdist).
- `-c /dev/null` skips the default configuration in `pyproject.toml`, which
  speeds up running a handful of tests.

## Design

- Keep changes small: ideally a pull request changes just a few files and no
  more than about 100 lines of production code.
- Keep code and documentation short and simple. Avoid complicated class
  hierarchies and boilerplate code.
- Check recipe settings that can be checked before loading the data when the
  tasks are created, so users find mistakes early. Do not process data or write
  files while creating the tasks.
- Preprocessor functions are plain functions that preferably just call iris.
  Code that is useful beyond ESMValCore belongs in iris. Do not add CMOR checks
  to preprocessor functions: their input has already been fixed and checked.
- Do not copy code from other projects unless its license is compatible with
  Apache 2.0.

## Code style

- Formatting and linting use [ruff](https://docs.astral.sh/ruff/) with nearly
  all rules enabled and a line length of 79; see `[tool.ruff]` in
  `pyproject.toml`. Type checks use mypy; other checks (codespell, yamllint,
  zizmor, ...) run via pre-commit. Run pre-commit before finishing a change.
- Write [numpy style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html)
  for public functions, classes, and modules. Private helpers only need a short
  docstring when their purpose is not obvious.
- Add type hints to new code and, as much as possible, to existing code that
  you edit, unless that would lead to many changes unrelated to the goal of the
  change. Type hints must be correct.
- Do not add `# noqa` or `# type: ignore` comments to silence a check unless
  there is no reasonable way to fix the underlying issue.
- Match the style, naming, and comment density of the surrounding code.

## Tests

- New and changed code must be covered by unit tests (`tests/unit/`), mirroring
  the package structure. Use `tests/integration/` for tests that exercise
  several components together.
- Tests use pytest; prefer plain functions, fixtures, `pytest.mark.parametrize`,
  and `pytest-mock`'s `mocker` fixture over `unittest` style.
- Do not change existing tests to make new code pass: having to change them
  may mean that existing features break. Flag such changes to the user.
- Tests that need internet access must be marked `@pytest.mark.online`; tests
  that use the ESMValTool sample data must be marked
  `@pytest.mark.use_sample_data`. New or changed preprocessor functions should
  preferably also be tested with the sample data in `tests/sample_data/`.
- `xfail_strict = true` is set, so an `xfail` test that unexpectedly passes
  counts as a failure.

## Documentation

- Update the documentation in `doc/` when behavior visible to users changes.
- In `.rst` files, keep lines at most 80 characters long and start each
  sentence on a new line.
- Public Python API only appears in the documentation if it is listed in
  `doc/api/`.
- For functions that compute scientific results, add comments with references
  to papers and formula numbers.
- The documentation must build without warnings: readthedocs builds it with
  `sphinx-build -W`, while `pixi run doc` does not turn warnings into errors.

## Common tasks

### Adding a preprocessor function

See [doc/develop/preprocessor_function.rst](doc/develop/preprocessor_function.rst).
In short:

1. Add the function to a suitable module in `esmvalcore/preprocessor/`. It takes
   an `iris.cube.Cube` (or list of cubes) as its first argument and returns the
   result.
2. Import it in `esmvalcore/preprocessor/__init__.py` and add it to `__all__`.
   The position in `__all__` sets the default order in which preprocessor
   steps are applied, so choose it carefully.
3. Add type hints for all arguments and document each one in the docstring.
   Recipe arguments are checked against the function signature; add any
   further validation of argument values to `esmvalcore/_recipe/check.py`.
4. Preserve laziness: lazy input data must give lazy output, real input data
   must give real output. Prefer calling iris over hand-written NumPy/Dask code.
5. Document it for recipe users in `doc/recipe/preprocessor.rst`.

### Changing the recipe format

When changing the structure of a recipe (sections, dataset and variable keys,
...), update the recipe validation (called from `esmvalcore/_recipe/check.py`)
together with the code that reads the recipe and the user documentation in
`doc/recipe/`.

### Fixing a dataset

Dataset fixes live in `esmvalcore/cmor/_fixes/<project>/<dataset>.py`. See
[doc/develop/fixing_data.rst](doc/develop/fixing_data.rst) and add tests under
`tests/integration/cmor/_fixes/`.

### Adding a derived variable

Add a module to `esmvalcore/preprocessor/_derive/` following
[doc/develop/derivation.rst](doc/develop/derivation.rst).

## Backward compatibility

ESMValCore has many users with their own recipes, so changes to the recipe
format, configuration, diagnostic script interface, or public Python API must
stay backward compatible where possible. To remove or change a public feature,
first deprecate it with an `ESMValCoreDeprecationWarning` (from
`esmvalcore.exceptions`), stating the version in which it will be removed (two
minor versions after the upcoming release), and label the pull request
`deprecated feature`. Flag any breaking change clearly to the user rather than
making it silently.

## Dependencies

When adding or removing a dependency, update both `[project] dependencies`
(PyPI) and `[tool.pixi.dependencies]` (conda-forge) in `pyproject.toml`, keep
them in sync, and run `pixi lock` to update `pixi.lock`. Tools that are only
needed for development, building the documentation, or running the tests go in
`[tool.pixi.feature.dev.dependencies]`, `[tool.pixi.feature.doc.dependencies]`,
or `[tool.pixi.feature.test.dependencies]` instead. New dependencies must be
actively maintained, be available on conda-forge, and have a license that is
compatible with Apache 2.0, as must their own dependencies. Supported Python
and dependency versions follow
[SPEC-0](https://scientific-python.org/specs/spec-0000/).

## Pull requests

- Follow the [AI policy](doc/ai_policy.rst). In particular: do not open pull
  requests autonomously (a human must review the code and submit it), do not
  write pull request descriptions, issue descriptions, or comments on behalf of
  the contributor, and remind the contributor to disclose which AI tools were
  used and for what.
- The changelog is generated from pull request titles and labels, so use a
  short, descriptive title that is understandable to users. Do not edit
  `doc/changelog.rst` by hand for regular changes.
- Add at least one label: `bug`, `deprecated feature`, `fix for dataset`,
  `preprocessor`, `cmor`, `api`, `testing`, `documentation`, or `enhancement`.
- Do not modify vendored data such as `esmvalcore/cmor/tables/`,
  `esmvalcore/preprocessor/shapefiles/`, or `esmvalcore/preprocessor/ne_masks/`
  unless that is the goal of the change.
