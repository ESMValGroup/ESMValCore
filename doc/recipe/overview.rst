.. _recipe_overview:

Overview
********

After ``config-user.yml``, the ``recipe.yml`` is the second file the user needs
to pass to ``esmvaltool`` as command line option, at each run time point.
Recipes contain the data and data analysis information and instructions needed
to run the diagnostic(s), as well as specific diagnostic-related instructions.

Broadly, recipes contain a general section summarizing the provenance and
functionality of the diagnostics, the datasets which need to be run, the
preprocessors that need to be applied, and the diagnostics which need to be run
over the preprocessed data. This information is provided to ESMValTool in four
main recipe sections: :ref:`Documentation <recipe_documentation>`, Datasets_,
Preprocessors_, and Diagnostics_, respectively.

.. _recipe_documentation:

Recipe section: ``documentation``
=================================

The documentation section includes:

- The recipe's author's user name (``authors``, matching the definitions in the
  :ref:`config-ref`)
- The recipe's maintainer's user name (``maintainer``, matching the definitions in the
  :ref:`config-ref`)
- The title of the recipe (``title``)
- A description of the recipe (``description``, written in MarkDown format)
- A list of scientific references (``references``, matching the definitions in
  the :ref:`config-ref`)
- the project or projects associated with the recipe (``projects``, matching
  the definitions in the :ref:`config-ref`)

For example, the documentation section of ``recipes/recipe_ocean_amoc.yml`` is
the following:

.. code-block:: yaml

    documentation:
      title: Atlantic Meridional Overturning Circulation (AMOC) and the drake passage current
      description: |
        Recipe to produce time series figures of the derived variable, the
        Atlantic meridional overturning circulation (AMOC).
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

   Note that all authors, projects, and references mentioned in the description
   section of the recipe need to be included in the (locally installed copy of the) file
   `esmvaltool/config-references.yml <https://github.com/ESMValGroup/ESMValTool/blob/main/esmvaltool/config-references.yml>`_,
   see :ref:`config-ref`.
   The author name uses the format: ``surname_name``. For instance, John
   Doe would be: ``doe_john``. This information can be omitted by new users
   whose name is not yet included in ``config-references.yml``.

.. _Datasets:

Recipe section: ``datasets``
============================

The ``datasets`` section includes dictionaries that, via key-value pairs, define standardized
data specifications:

- dataset name (key ``dataset``, value e.g. ``MPI-ESM-LR`` or ``UKESM1-0-LL``).
- project (key ``project``, value ``CMIP5`` or ``CMIP6`` for CMIP data,
  ``OBS`` for observational data, ``ana4mips`` for ana4mips data,
  ``obs4MIPs`` for obs4MIPs data, ``ICON`` for ICON data).
- experiment (key ``exp``, value e.g. ``historical``, ``amip``, ``piControl``,
  ``rcp85``).
- mip (for CMIP data, key ``mip``, value e.g. ``Amon``, ``Omon``, ``LImon``).
- ensemble member (key ``ensemble``, value e.g. ``r1i1p1``, ``r1i1p1f1``).
- sub-experiment id (key ``sub_experiment``, value e.g. ``s2000``,
  ``s(2000:2002)``, for DCPP data only).
- time range (e.g. key-value ``start_year: 1982``, ``end_year: 1990``).
  Please note that `yaml`_ interprets numbers with a leading ``0`` as octal
  numbers, so we recommend to avoid them. For example, use ``128`` to specify
  the year 128 instead of ``0128``.
  Alternatively, the time range can be specified in `ISO 8601 format
  <https://en.wikipedia.org/wiki/ISO_8601>`_, for both dates and periods.
  In addition, wildcards (``'*'``) are accepted, which allow the selection of
  the first available year for each individual dataset (when used as a starting
  point) or the last available year (when used as an ending point).
  The starting point and end point must be separated with ``/`` (e.g. key-value
  ``timerange: '1982/1990'``).
  More examples are given :ref:`here <timerange_examples>`.
- model grid (native grid ``grid: gn`` or regridded grid ``grid: gr``, for
  CMIP6 data only).

For example, a datasets section could be:

.. code-block:: yaml

    datasets:
      - {dataset: CanESM2, project: CMIP5, exp: historical, ensemble: r1i1p1, start_year: 2001, end_year: 2004}
      - {dataset: UKESM1-0-LL, project: CMIP6, exp: historical, ensemble: r1i1p1f2, start_year: 2001, end_year: 2004, grid: gn}
      - {dataset: ACCESS-CM2, project: CMIP6, exp: historical, ensemble: r1i1p1f2, timerange: 'P5Y/*', grid: gn}
      - {dataset: EC-EARTH3, alias: custom_alias, project: CMIP6, exp: historical, ensemble: r1i1p1f1, start_year: 2001, end_year: 2004, grid: gn}
      - {dataset: CMCC-CM2-SR5, project: CMIP6, exp: historical, ensemble: r1i1p1f1, timerange: '2001/P10Y', grid: gn}
      - {dataset: HadGEM3-GC31-MM, project: CMIP6, exp: dcppA-hindcast, ensemble: r1i1p1f1, sub_experiment: s2000, grid: gn, start_year: 2000, end_year, 2002}
      - {dataset: BCC-CSM2-MR, project: CMIP6, exp: dcppA-hindcast, ensemble: r1i1p1f1, sub_experiment: s2000, grid: gn, timerange: '*'}

It is possible to define the experiment as a list to concatenate two experiments.
Here it is an example concatenating the `historical` experiment with `rcp85`

.. code-block:: yaml

    datasets:
      - {dataset: CanESM2, project: CMIP5, exp: [historical, rcp85], ensemble: r1i1p1, start_year: 2001, end_year: 2004}

It is also possible to define the ensemble as a list when the two experiments have different ensemble names.
In this case, the specified datasets are concatenated into a single cube:

.. code-block:: yaml

    datasets:
      - {dataset: CanESM2, project: CMIP5, exp: [historical, rcp85], ensemble: [r1i1p1, r1i2p1], start_year: 2001, end_year: 2004}

ESMValTool also supports a simplified syntax to add multiple ensemble members from the same dataset.
In the ensemble key, any element in the form `(x:y)` will be replaced with all numbers from x to y (both inclusive),
adding a dataset entry for each replacement. For example, to add ensemble members r1i1p1 to r10i1p1
you can use the following abbreviated syntax:

.. code-block:: yaml

    datasets:
      - {dataset: CanESM2, project: CMIP5, exp: historical, ensemble: "r(1:10)i1p1", start_year: 2001, end_year: 2004}

It can be included multiple times in one definition. For example, to generate the datasets definitions
for the ensemble members r1i1p1 to r5i1p1 and from r1i2p1 to r5i1p1 you can use:

.. code-block:: yaml

    datasets:
      - {dataset: CanESM2, project: CMIP5, exp: historical, ensemble: "r(1:5)i(1:2)p1", start_year: 2001, end_year: 2004}

Please, bear in mind that this syntax can only be used in the ensemble tag.
Also, note that the combination of multiple experiments and ensembles, like
exp: [historical, rcp85], ensemble: [r1i1p1, "r(2:3)i1p1"] is not supported and will raise an error.

The same simplified syntax can be used to add multiple sub-experiment ids:

.. code-block:: yaml

    datasets:
      - {dataset: MIROC6, project: CMIP6, exp: dcppA-hindcast, ensemble: r1i1p1f1, sub_experiment: s(2000:2002), grid: gn, start_year: 2003, end_year: 2004}

.. _timerange_examples:

When using the ``timerange`` tag to specify the start and end points, possible values can be as follows:

  - A start and end point specified with a resolution up to seconds (YYYYMMDDThhmmss)
    * ``timerange: '1980/1982'``. Spans from 01/01/1980 to 31/12/1980.
    * ``timerange: '198002/198205'``. Spans from 01/02/1980 to 31/05/1982.
    * ``timerange: '19800302/19820403'``. Spans from 02/03/1980 to 03/04/1982.
    * ``timerange: '19800504T100000/19800504T110000'``. Spans from 04/05/1980 at 10h to 11h.

  - A start point or end point, and a relative period with a resolution up to second (P[n]Y[n]M[n]DT[n]H[n]M[n]S).
    * ``timerange: '1980/P5Y'``. Starting from 01/01/1980, spans 5 years.
    * ``timerange: 'P2Y5M/198202``. Ending at 28/02/1982, spans 2 years and 5 months.

  - A wildcard to load all available years, the first available start point or the last available end point.
    * ``timerange: '*'``. Finds all available years.
    * ``timerange: '*/1982``. Finds first available point, spans to 31/12/1982.
    * ``timerange: '*/P6Y``. Finds first available point, spans 6 years from it.
    * ``timerange: '198003/*``. Starting from 01/03/1980, spans until the last available point.
    * ``timerange: 'P5M/*``. Finds last available point, spans 5 months backwards from it.


Note that this section is not required, as datasets can also be provided in the
Diagnostics_ section.

.. _`yaml`: https://yaml.org/refcard.html

.. _Preprocessors:

Recipe section: ``preprocessors``
=================================

The preprocessor section of the recipe includes one or more preprocessors, each
of which may call the execution of one or several preprocessor functions.

Each preprocessor section includes:

- A preprocessor name (any name, under ``preprocessors``);
- A list of preprocessor steps to be executed (choose from the API);
- Any or none arguments given to the preprocessor steps;
- The order that the preprocessor steps are applied can also be specified using
  the ``custom_order`` preprocessor function.

The following snippet is an example of a preprocessor named ``prep_map`` that
contains multiple preprocessing steps (:ref:`Horizontal regridding` with two
arguments, :ref:`Time operations` with no arguments (i.e., calculating the
average over the time dimension) and :ref:`Multi-model statistics` with two
arguments):

.. code-block:: yaml

    preprocessors:
      prep_map:
        regrid:
          target_grid: 1x1
          scheme: linear
        climate_statistics:
          operator: mean
        multi_model_statistics:
          span: overlap
          statistics: [mean]

.. note::

   In this case no ``preprocessors`` section is needed the workflow will apply
   a ``default`` preprocessor consisting of only basic operations like: loading
   data, applying CMOR checks and fixes (:ref:`CMOR check and dataset-specific
   fixes`) and saving the data to disk.

Preprocessor operations will be applied using the default order
as listed in :ref:`preprocessor_functions`.
Preprocessor tasks can be set to run in the order they are listed in the recipe
by adding ``custom_order: true`` to the preprocessor definition.

.. _Diagnostics:

Recipe section: ``diagnostics``
===============================

The diagnostics section includes one or more diagnostics. Each diagnostic
section will include:

- the variable(s) to preprocess, including the preprocessor to be applied to each variable;
- the diagnostic script(s) to be run;
- a description of the diagnostic and lists of themes and realms that it applies to;
- an optional ``additional_datasets`` section.
- an optional ``title`` and ``description``, used to generate the title and description
  of the ``index.html`` output file.

.. _tasks:

The diagnostics section defines tasks
-------------------------------------
The diagnostic section(s) define the tasks that will be executed when running the recipe.
For each variable a preprocessing task will be defined and for each diagnostic script a
diagnostic task will be defined. If variables need to be derived
from other variables, a preprocessing task for each of the variables
needed to derive that variable will be defined as well. These tasks can be viewed
in the main_log_debug.txt file that is produced every run. Each task has a unique
name that defines the subdirectory where the results of that task are stored. Task
names start with the name of the diagnostic section followed by a '/' and then
the name of the variable section for a preprocessing task or the name of the diagnostic
script section for a diagnostic task.

A (simplified) example diagnostics section could look like

.. code-block:: yaml

  diagnostics:
    diagnostic_name:
      title: Air temperature tutorial diagnostic
      description: A longer description can be added here.
      themes:
        - phys
      realms:
        - atmos
      variables:
        variable_name:
          short_name: ta
          preprocessor: preprocessor_name
          mip: Amon
      scripts:
        script_name:
          script: examples/diagnostic.py


Note that the example recipe above contains a single diagnostic section
called ``diagnostic_name`` and will result in two tasks:

- a preprocessing task called ``diagnostic_name/variable_name`` that will preprocess
  air temperature data for each dataset in the Datasets_ section of the recipe (not shown).
- a diagnostic task called ``diagnostic_name/script_name``

The path to the script provided in the ``script`` option should be
either the absolute path to the script, or the path relative to the
``esmvaltool/diag_scripts`` directory.

Depending on the installation configuration, you may get an error of
"file does not exist" when the system tries to run the diagnostic script
using relative paths. If this happens, use an absolute path instead.

Note that the script should either have the extension for a supported language,
i.e. ``.py``, ``.R``, ``.ncl``, or ``.jl`` for Python, R, NCL, and Julia diagnostics
respectively, or be executable if it is written in any other language.

.. _ancestor-tasks:

Ancestor tasks
--------------
Some tasks require the result of other tasks to be ready before they can start,
e.g. a diagnostic script needs the preprocessed variable data to start. Thus
each tasks has zero or more ancestor tasks. By default, each diagnostic task
in a diagnostic section has all variable preprocessing tasks in that same section
as ancestors. However, this can be changed using the ``ancestors`` keyword. Note
that wildcard expansion can be used to define ancestors.

.. code-block:: yaml

  diagnostics:
    diagnostic_1:
      variables:
        airtemp:
          short_name: ta
          preprocessor: preprocessor_name
          mip: Amon
      scripts:
        script_a:
          script: diagnostic_a.py
    diagnostic_2:
      variables:
        precip:
          short_name: pr
          preprocessor: preprocessor_name
          mip: Amon
      scripts:
        script_b:
          script: diagnostic_b.py
          ancestors: [diagnostic_1/script_a, precip]


The example recipe above will result in four tasks:

- a preprocessing task called ``diagnostic_1/airtemp``
- a diagnostic task called ``diagnostic_1/script_a``
- a preprocessing task called ``diagnostic_2/precip``
- a diagnostic task called ``diagnostic_2/script_b``

the preprocessing tasks do not have any ancestors, while the diagnostic_a.py
script will receive the preprocessed air temperature data
(has ancestor ``diagnostic_1/airtemp``) and the diagnostic_b.py
script will receive the results of diagnostic_a.py and the preprocessed precipitation
data (has ancestors ``diagnostic_1/script_a`` and ``diagnostic_2/precip``).

Task priority
-------------
Tasks are assigned a priority, with tasks appearing earlier on in the recipe
getting higher priority. The tasks will be executed sequentially or in parellel,
depending on the setting of ``max_parallel_tasks`` in the :ref:`user configuration file`.
When there are fewer than ``max_parallel_tasks`` running, tasks will be started
according to their priority. For obvious reasons, only tasks that are not waiting for
ancestor tasks can be started. This feature makes it possible to
reduce the processing time of recipes with many tasks, by placing tasks that
take relatively long near the top of the recipe. Of course this only works when
settings ``max_parallel_tasks`` to a value larger than 1. The current priority
and run time of individual tasks can be seen in the log messages shown when
running the tool (a lower number means higher priority).

Variable and dataset definitions
--------------------------------
To define a variable/dataset combination that corresponds to an actual
variable from a dataset, the keys in each variable section
are combined with the keys of each dataset definition. If two versions of the same
key are provided, then the key in the datasets section will take precedence
over the keys in variables section. For many recipes it makes more sense to
define the ``start_year`` and ``end_year`` items in the variable section,
because the diagnostic script assumes that all the data has the same time
range.

Variable short names usually do not change between datasets supported by
ESMValCore, as they are usually changed to match CMIP. Nevertheless, there are
small changes in variable names in CMIP6 with respect to CMIP5 (i.e. sea ice
concentration changed from ``sic`` to ``siconc``). ESMValCore is aware of some
of them and can do the automatic translation when needed. It will even do the
translation in the preprocessed file so the diagnostic does not have to deal
with this complexity, setting the short name in all files to match the one used
by the recipe. For example, if ``sic`` is requested, ESMValCore will
find ``sic`` or ``siconc`` depending on the project, but all preprocessed files
while use ``sic`` as their short_name. If the recipe requested ``siconc``, the
preprocessed files will be identical except that they will use the short_name
``siconc`` instead.

Diagnostic and variable specific datasets
-----------------------------------------
The ``additional_datasets`` option can be used to add datasets beyond those
listed in the Datasets_ section. This is useful if specific datasets need to
be used only by a specific diagnostic or variable, i.e. it can be added both
at diagnostic level, where it will apply to all variables in that diagnostic
section or at individual variable level. For example, this can be a good way
to add observational datasets, which are usually variable-specific.

Running a simple diagnostic
---------------------------
The following example, taken from ``recipe_ocean_example.yml``, shows a
diagnostic named `diag_map`, which loads the temperature at the ocean surface
between the years 2001 and 2003 and then passes it to the ``prep_map``
preprocessor. The result of this process is then passed to the ocean diagnostic
map script, ``ocean/diagnostic_maps.py``.

.. code-block:: yaml

  diagnostics:

    diag_map:
      title: Global Ocean Surface regridded temperature map
      description: Add a longer description here.
      variables:
        tos: # Temperature at the ocean surface
          preprocessor: prep_map
          start_year: 2001
          end_year: 2003
      scripts:
        Global_Ocean_Surface_regrid_map:
          script: ocean/diagnostic_maps.py

Passing arguments to a diagnostic script
----------------------------------------
The diagnostic script section(s) may include custom arguments that can be used by
the diagnostic script; these arguments are stored at runtime in a dictionary
that is then made available to the diagnostic script via the interface link,
independent of the language the diagnostic script is written in. Here is an
example of such groups of arguments:

.. code-block:: yaml

    scripts:
      autoassess_strato_test_1: &autoassess_strato_test_1_settings
        script: autoassess/autoassess_area_base.py
        title: "Autoassess Stratosphere Diagnostic Metric MPI-MPI"
        area: stratosphere
        control_model: MPI-ESM-LR
        exp_model: MPI-ESM-MR
        obs_models: [ERA-Interim]  # list to hold models that are NOT for metrics but for obs operations
        additional_metrics: [ERA-Interim, inmcm4]  # list to hold additional datasets for metrics

In this example, apart from specifying the diagnostic script ``script:
autoassess/autoassess_area_base.py``, we pass a suite of parameters to be used
by the script (``area``, ``control_model`` etc). These parameters are stored in
key-value pairs in the diagnostic configuration file, an interface file that
can be used by importing the ``run_diagnostic`` utility:

.. code-block:: python

   from esmvaltool.diag_scripts.shared import run_diagnostic

   # write the diagnostic code here e.g.
   def run_some_diagnostic(my_area, my_control_model, my_exp_model):
       """Diagnostic to be run."""
       if my_area == 'stratosphere':
           diag = my_control_model / my_exp_model
           return diag

   def main(cfg):
       """Main diagnostic run function."""
       my_area = cfg['area']
       my_control_model = cfg['control_model']
       my_exp_model = cfg['exp_model']
       run_some_diagnostic(my_area, my_control_model, my_exp_model)

   if __name__ == '__main__':

       with run_diagnostic() as config:
           main(config)

This way a lot of the optional arguments necessary to a diagnostic are at the
user's control via the recipe.

Running your own diagnostic
---------------------------
If the user wants to test a newly-developed ``my_first_diagnostic.py`` which
is not yet part of the ESMValTool diagnostics library, he/she do it by passing
the absolute path to the diagnostic:

.. code-block:: yaml

  diagnostics:

    myFirstDiag:
      title: Let's do some science!
      description: John Doe wrote a funny diagnostic
      variables:
        tos: # Temperature at the ocean surface
          preprocessor: prep_map
          start_year: 2001
          end_year: 2003
      scripts:
        JoeDiagFunny:
          script: /home/users/john_doe/esmvaltool_testing/my_first_diagnostic.py

This way the user may test a new diagnostic thoroughly before committing to the
GitHub repository and including it in the ESMValTool diagnostics library.

Re-using parameters from one ``script`` to another
--------------------------------------------------
Due to ``yaml`` features it is possible to recycle entire diagnostics sections
for use with other diagnostics. Here is an example:

.. code-block:: yaml

    scripts:
      cycle: &cycle_settings
        script: perfmetrics/main.ncl
        plot_type: cycle
        time_avg: monthlyclim
      grading: &grading_settings
        <<: *cycle_settings
        plot_type: cycle_latlon
        calc_grading: true
        normalization: [centered_median, none]

In this example the hook ``&cycle_settings`` can be used to pass the ``cycle:``
parameters to ``grading:`` via the shortcut ``<<: *cycle_settings``.
