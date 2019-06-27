.. _recipe:

******
Recipe
******

Overview
========

After ``config-user.yml``, the ``recipe.yml`` file the second the user inputted file
to ESMValTool at each run time point. Rceipes contain the data and data analysis
information and instructions needed to run the diagnostic(s), as well as specific
diagnostic-related instructions.

Broadly, recipes contain information on the user who wrote the
recipe file, the datasets which need to be run, the preprocessors that need to be
applied, and the diagnostics which need to be run over the preprocessed data.
This information is provided to ESMValTool in for main recipe sections:
Documentation_, Datasets_, Preprocessors_ and Diagnostics_,
respectively.

.. _Documentation:

Recipe section: ``documentation``
=================================

The documentation section includes:

- The recipe's author's user name (``authors``, as they appaer in ``config-references.yml`` config-ref_)
- A description of the recipe (``description``, written in MarkDown format)
- The user name of the maintainer (``maintainer``, as they appaer in ``config-references.yml`` config-ref_)
- A list of scientific references (``references`` , as they appaer in ``config-references.yml`` config-ref_)
- the project or projects associated with the recipe (``projects``, as they appaer in ``config-references.yml`` config-ref_)

For example, please see the documentation section from the recipe:
``recipes/recipe_ocean_amoc.yml``:

.. code-block:: yaml

    documentation:
      description: |
        Recipe to produce time series figures of the derived variable, the
        Atlantic meriodinal overturning circulation (AMOC).
        This recipe also produces transect figures of the stream functions for
        the years 2001-2004.

      authors:
        - demo_le

      maintainer:
        - demo_le

      references:
        - demora2018gmd

      projects:
        - ukesm

.. note::

   **Information from config-references.yml**

   Note that the authors, projects, and references will need to be included in the
   ``config-references.yml`` file. The author name uses the format:
   ``surname_name``. For instance, Joe Pesci would be: ``authors: - pesci_joe``.
   Also note that Joe Pesci does not appreciate you calling him ``funny``.
   For a first-time user that does not yet have their name added to ``config-references.yml``
   a run of an already-made recipe or running with no author name is possible.

.. _Datasets:

Recipe section: ``datasets``
============================

The ``datasets`` section includes dictionaries that, via key-value pairs, define standardized
data specifications:

- dataset name (key ``dataset``, value e.g. ``MPI-ESM-LR`` or ``UKESM1-0-LL``)
- project (key ``project``, value ``CMIP5`` or ``CMIP6`` for CMIP data,
  ``OBS`` for observational data, ``ana4mips`` for ana4mips data,
  ``obs4mips`` for obs4mips data, ``EMAC`` for EMAC data)
- experiment (key ``exp``, value e.g. ``historical``, ``amip``, ``piControl``, ``RCP8.5``)
- mip (for CMIP data, key ``mip``, value e.g. ``Amon``, ``Omon``, ``LImon``)
- ensemble member (key ``ensemble``, value e.g. ``r1i1p1``, ``r1i1p1f1``)
- time range (e.g. key-value ``start_year: 1982``, ``end_year: 1990``)
- model grid (native grid ``grid: gn`` or regridded grid ``grid: gr``, for CMIP6 data only).

For example, a datasets section could be:

.. code-block:: yaml

    datasets:
      - {dataset: CanESM2, project: CMIP5, exp: historical, ensemble: r1i1p1, start_year: 2001, end_year: 2004}
      - {dataset: UKESM1-0-LL, project: CMIP6, exp: historical, ensemble: r1i1p1f2, start_year: 2001, end_year: 2004, grid: gn}


Note that this section is not required, as datasets can also be provided in the
Diagnostics_ section.

.. _Preprocessors:

Recipe section: ``preprocessors``
=================================

The preprocessor section of the recipe includes one or more preprocesors, each
of which may execute one or several preprocessor steps.

Each preprocessor section includes:

- A preprocessor name (any name, one yaml tab (2 spaces) under ``preprocessors``);
- A list of preprocesor steps to be executed (choose from the API, each one yaml tab (2 spaces) under preprocessor name)
- Any or none arguments given to the preprocessor steps (one yaml tab (2 spaces) under preprocessor step name)
- The order that the preprocesor steps are applied can also be specified using the ``custom_order`` preprocesor function.

The following snippet is an example of a preprocessor named ``prep_map`` that contains
multiple preprocessing steps (regrid_ with two arguments, time_average_ with no arguments
and multi_model_statistics_ with two arguments):

.. code-block:: yaml

    preprocessors:
      prep_map:
        regrid:
          target_grid: 1x1
          scheme: linear
        time_average:
        multi_model_statistics:
          span: overlap
          statistics: [mean ]

.. note::

   What if no preprocessor is needed?

   In this case no ``preprocessors`` section is needed;
   the workflow will apply a ``default`` preprocessor consisting of only
   basic operations like: loading data, applying CMOR checks and fixes (cmor-checks-fixes_)
   and saving the data to disk (if needed).

.. _Diagnostics:

Recipe section: ``diagnostics``
===============================

The diagnostics section includes one or more diagnostics. Each diagnostics will
have:

- A list of which variables to load
- A description of the variables (optional)
- Which preprocessor to apply to each variable
- The script to run
- The diagnostics can also include an optional ``additional_datasets`` section.

The ``additional_datasets`` can add datasets beyond those listed in the the
`Datasets`_ section. This is useful if specific datasets need to be linked with
a specific diagnostics. The addition datasets can be used to add variable
specific datasets. This is also a good way to add observational datasets can be
added to the diagnostic.

The following example, taken from recipe_ocean_example.yml, shows a diagnostic
named `diag_map`, which loads the temperature at the ocean surface between
the years 2001 and 2003 and then passes it to the prep_map preprocessor.
The result of this process is then passed to the ocean diagnostic map scipt,
``ocean/diagnostic_maps.py``.

.. code-block:: yaml

    diagnostics:

    diag_map:
      description: Global Ocean Surface regridded temperature map
      variables:
        tos: # Temperature at the ocean surface
          preprocessor: prep_map
          start_year: 2001
          end_year: 2003
      scripts:
        Global_Ocean_Surface_regrid_map:
          script: ocean/diagnostic_maps.py

To define a variable/dataset combination, the keys in the diagnostic section
are combined with the keys from datasets section. If two versions of the same
key are provided, then the key in the datasets section will take precedence
over the keys in variables section. For many recipes it makes more sense to
define the ``start_year`` and ``end_year`` items in the variable section, because the
diagnostic script assumes that all the data has the same time range.

Note that the path to the script provided in the `script` option should be
either:

1. the absolute path to the script.
2. the path relative to the ``esmvaltool/diag_scripts`` directory.


As mentioned above, the datasets are provided in the `Diagnostics`_ section
in this section. However, they could also be included in the `Datasets`_
section.


Brief introduction to YAML
==========================

While .yaml is a relatively common format, maybe users may not have
encountered this language before. The key information about this format is:

- Yaml is a human friendly markup language.
- Yaml is commonly used for configuration files.
- the syntax is relatively straightforward
- Indentation matters a lot (like python)!
- yaml is case sensitive
- A yml tutorial is available here: https://learnxinyminutes.com/docs/yaml/
- A yml quick reference card is available here: https://yaml.org/refcard.html
- ESMValTool uses the yamllint linter tool: http://www.yamllint.com
