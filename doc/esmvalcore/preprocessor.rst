.. _preprocessor:

************
Preprocessor
************

In this section, each of the preprocessor modules is described in detail
following the default order in which they are applied:

* :ref:`Variable derivation`
* :ref:`CMOR check and dataset-specific fixes`
* :ref:`Vertical interpolation`
* :ref:`Land/Sea/Ice masking`
* :ref:`Horizontal regridding`
* :ref:`Masking of missing values`
* :ref:`Multi-model statistics`
* :ref:`Time operations`
* :ref:`Area operations`
* :ref:`Volume operations`
* :ref:`Detrend`
* :ref:`Unit conversion`

Overview
========

..
   ESMValTool is a modular ``Python 3.6+`` software package possesing capabilities
   of executing a large number of diagnostic routines that can be written in a
   number of programming languages (Python, NCL, R, Julia). The modular nature
   benefits the users and developers in different key areas: a new feature
   developed specifically for version 2.0 is the preprocessing core  or the
   preprocessor (esmvalcore) that executes the bulk of standardized data
   operations and is highly optimized for maximum performance in data-intensive
   tasks. The main objective of the preprocessor is to integrate as many
   standardizable data analysis functions as possible so that the diagnostics can
   focus on the specific scientific tasks they carry. The preprocessor is linked
   to the diagnostics library and the diagnostic execution is seamlessly performed
   after the preprocessor has completed its steps. The benefit of having a
   preprocessing unit separate from the diagnostics library include:

   * ease of integration of new preprocessing routines;
   * ease of maintenance (including unit and integration testing) of existing
     routines;
   * a straightforward manner of importing and using the preprocessing routines as
     part  of the overall usage of the software and, as a special case, the use
     during diagnostic execution;
   * shifting the effort for the scientific diagnostic developer from implementing
     both standard and diagnostic-specific functionalities to allowing them to
     dedicate most of the effort to developing scientifically-relevant diagnostics
     and metrics;
   * a more strict code review process, given the smaller code base than for
     diagnostics.

The ESMValTool preprocessor can be used to perform a broad range of operations
on the input data before diagnostics or metrics are applied. The preprocessor
performs these operations in a centralized, documented and efficient way, thus
reducing the data processing load on the diagnostics side.

Each of the preprocessor operations is written in a dedicated python module and
all of them receive and return an Iris `cube
<https://scitools.org.uk/iris/docs/v2.0/iris/iris/cube.html>`_ , working
sequentially on the data with no interactions between them. The order in which
the preprocessor operations is applied is set by default to minimize
the loss of information due to, for example, temporal and spatial subsetting or
multi-model averaging. Nevertheless, the user is free to change such order to
address specific scientific requirements, but keeping in mind that some
operations must be necessarily performed in a specific order. This is the case,
for instance, for multi-model statistics, which required the model to be on a
common grid and therefore has to be called after the regridding module.


.. _Variable derivation:

Variable derivation
===================
The variable derivation module allows to derive variables which are not in the
CMIP standard data request using standard variables as input. The typical use
case of this operation is the evaluation of a variable which is only available
in an observational dataset but not in the models. In this case a derivation
function is provided by the ESMValTool in order to calculate the variable and
perform the comparison. For example, several observational datasets deliver
total column ozone as observed variable (`toz`), but CMIP models only provide
the ozone 3D field. In this case, a derivation function is provided to
vertically integrate the ozone and obtain total column ozone for direct
comparison with the observations.

To contribute a new derived variable, it is also necessary to define a name for
it and to provide the corresponding CMOR table. This is to guarantee the proper
metadata definition is attached to the derived data. Such custom CMOR tables
are collected as part of the `ESMValCore package
<https://github.com/ESMValGroup/ESMValCore>`_. By default, the variable
derivation will be applied only if the variable is not already available in the
input data, but the derivation can be forced by setting the appropriate flag.

.. code-block:: yaml

  variables:
    toz:
      derive: true
      force_derivation: false

The required arguments for this module are two boolean switches:

* ``derive``: activate variable derivation
* ``force_derivation``: force variable derivation even if the variable is
  directly available in the input data.

See also :func:`esmvalcore.preprocessor.derive`. To get an overview on
derivation scripts and how to implement new ones, please go to
:ref:`derivation`.


.. _CMOR check and dataset-specific fixes:

CMORization and dataset-specific fixes
======================================

Data checking
-------------

Data preprocessed by ESMValTool is automatically checked against its cmor
definition. To reduce the impact of this check while maintaing it as realiable
as possible, it is split in two parts: one will check the metadata and will
be done just after loading and concatenating the data and the other one will
check the data itself and will be applied after all extracting operations are
applied to reduce the amount of data to process.

Checks include, but are not limited to:

   - Requested coordinates are present and comply with their definition.
   - Correctness of variable names, units and other metadata.
   - Compliance with the valid minimum and maximum values allowed if defined.

The most relevant (i.e. a missing coordinate) will raise an error while others
(i.e an incorrect long name) will be reported as a warning.

Some of those issues will be fixed automatically by the tool, including the
following:

    - Incorrect standard or long names.
    - Incorrect units, if they can be converted to the correct ones.
    - Direction of coordinates.
    - Automatic clipping of longitude to 0 - 360 interval.


Dataset specific fixes
----------------------

Sometimes, the checker will detect errors that it can not fix by itself.
ESMValTool deals with those issues by applying specific fixes for those
datasets that require them. Fixes are applied at three different preprocessor
steps:

    - fix_file: apply fixes directly to a copy of the file. Copying the files
      is costly, so only errors that prevent Iris to load the file are fixed
      here. See :func:`esmvalcore.preprocessor.fix_file`

    - fix_metadata: metadata fixes are done just before concatenating the cubes
      loaded from different files in the final one. Automatic metadata fixes
      are also applied at this step. See
      :func:`esmvalcore.preprocessor.fix_metadata`

    - fix_data: data fixes are applied before starting any operation that will
      alter the data itself. Automatic data fixes are also applied at this step.
      See :func:`esmvalcore.preprocessor.fix_data`

To get an overview on data fixes and how to implement new ones, please go to
:ref:`fixing_data`.


.. _Vertical interpolation:

Vertical interpolation
======================
Vertical level selection is an important aspect of data preprocessing since it
allows the scientist to perform a number of metrics specific to certain levels
(whether it be air pressure or depth, e.g. the Quasi-Biennial-Oscillation (QBO)
u30 is computed at 30 hPa). Dataset native vertical grids may not come with the
desired set of levels, so an interpolation operation will be needed to regrid
the data vertically. ESMValTool can perform this vertical interpolation via the
``extract_levels`` preprocessor. Level extraction may be done in a number of
ways.

Level extraction can be done at specific values passed to ``extract_levels`` as
``levels:`` with its value a list of levels (note that the units are
CMOR-standard, Pascals (Pa)):

.. code-block:: yaml

    preprocessors:
      preproc_select_levels_from_list:
        extract_levels:
          levels: [100000., 50000., 3000., 1000.]
          scheme: linear

It is also possible to extract the CMIP-specific, CMOR levels as they appear in
the CMOR table, e.g. ``plev10`` or ``plev17`` or ``plev19`` etc:

.. code-block:: yaml

    preprocessors:
      preproc_select_levels_from_cmip_table:
        extract_levels:
          levels: {cmor_table: CMIP6, coordinate: plev10}
          scheme: nearest

Of good use is also the level extraction with values specific to a certain
dataset, without the user actually polling the dataset of interest to find out
the specific levels: e.g. in the example below we offer two alternatives to
extract the levels and vertically regrid onto the vertical levels of
``ERA-Interim``:

.. code-block:: yaml

    preprocessors:
      preproc_select_levels_from_dataset:
        extract_levels:
          levels: ERA-Interim
          # This also works, but allows specifying the pressure coordinate name
          # levels: {dataset: ERA-Interim, coordinate: air_pressure}
          scheme: linear_horizontal_extrapolate_vertical

* See also :func:`esmvalcore.preprocessor.extract_levels`.
* See also :func:`esmvalcore.preprocessor.get_cmor_levels`.

.. note::

   For both vertical and horizontal regridding one can control the
   extrapolation mode when defining the interpolation scheme. Controlling the
   extrapolation mode allows us to avoid situations where extrapolating values
   makes little physical sense (e.g. extrapolating beyond the last data point).
   The extrapolation mode is controlled by the `extrapolation_mode`
   keyword. For the available interpolation schemes available in Iris, the
   extrapolation_mode keyword must be one of:

        * ``extrapolate``: the extrapolation points will be calculated by
	  extending the gradient of the closest two points;
        * ``error``: a ``ValueError`` exception will be raised, notifying an
	  attempt to extrapolate;
        * ``nan``: the extrapolation points will be be set to NaN;
        * ``mask``: the extrapolation points will always be masked, even if the
	  source data is not a ``MaskedArray``; or
        * ``nanmask``: if the source data is a MaskedArray the extrapolation
	  points will be masked, otherwise they will be set to NaN.


.. _masking:

Masking
=======

Introduction to masking
-----------------------

Certain metrics and diagnostics need to be computed and performed on specific
domains on the globe. The ESMValTool preprocessor supports filtering
the input data on continents, oceans/seas and ice. This is achived by masking
the model data and keeping only the values associated with grid points that
correspond to, e.g., land, ocean or ice surfaces, as specified by the
user. Where possible, the masking is realized using the standard mask files
provided together with the model data as part of the CMIP data request (the
so-called fx variable). In the absence of these files, the Natural Earth masks
are used: although these are not model-specific, they represent a good
approximation since they have a much higher resolution than most of the models
and they are regularly updated with changing geographical features.

.. _land/sea/ice masking:

Land-sea masking
----------------

In ESMValTool, land-sea-ice masking can be done in two places: in the
preprocessor, to apply a mask on the data before any subsequent preprocessing
step and before running the diagnostic, or in the diagnostic scripts
themselves. We present both these implementations below.

To mask out a certain domain (e.g., sea) in the preprocessor,
``mask_landsea`` can be used:

.. code-block:: yaml

    preprocessors:
      preproc_mask:
        mask_landsea:
          mask_out: sea

and requires only one argument: ``mask_out``: either ``land`` or ``sea``.

The preprocessor automatically retrieves the corresponding mask (``fx: stfof``
in this case) and applies it so that sea-covered grid cells are set to
missing. Conversely, it retrieves the ``fx: sftlf`` mask when land needs to be
masked out, respectively. If the corresponding fx file is not found (which is
the case for some models and almost all observational datasets), the
preprocessor attempts to mask the data using Natural Earth mask files (that are
vectorized rasters). As mentioned above, the spatial resolution of the the
Natural Earth masks are much higher than any typical global model (10m for
land and 50m for ocean masks).

See also :func:`esmvalcore.preprocessor.mask_landsea`.

Ice masking
-----------

Note that for masking out ice sheets, the preprocessor uses a different
function, to ensure that both land and sea or ice can be masked out without
losing generality. To mask ice out, ``mask_landseaice`` can be used:

.. code-block:: yaml

  preprocessors:
    preproc_mask:
      mask_landseaice:
        mask_out: ice

and requires only one argument: ``mask_out``: either ``landsea`` or ``ice``.

As in the case of ``mask_landsea``, the preprocessor automatically retrieves
the ``fx_files: [sftgif]`` mask.

See also :func:`esmvalcore.preprocessor.mask_landseaice`.

Mask files
----------

At the core of the land/sea/ice masking in the preprocessor are the mask files
(whether it be fx type or Natural Earth type of files); these files (bar
Natural Earth) can be retrived and used in the diagnostic phase as well. By
specifying the ``fx_files:`` key in the variable in diagnostic in the recipe,
and populating it with a list of desired files e.g.:

.. code-block:: yaml

    variables:
      ta:
        preprocessor: my_masking_preprocessor
          fx_files: [sftlf, sftof, sftgif, areacello, areacella]

Such a recipe will automatically retrieve all the ``fx_files: [sftlf, sftof,
sftgif, areacello, areacella]``-type fx files for each of the variables they
are needed for and then, in the diagnostic phase, these mask files will be
available for the developer to use them as they need to. The `fx_files`
attribute of the big `variable` nested dictionary that gets passed to the
diagnostic is, in turn, a dictionary on its own, and members of it can be
accessed in the diagnostic through a simple loop over the ``config`` diagnostic
variable items e.g.:

.. code-block::

    for filename, attributes in config['input_data'].items():
        sftlf_file = attributes['fx_files']['sftlf']
        areacello_file = attributes['fx_files']['areacello']

.. _masking of missing values:

Missing values masks
--------------------

Missing (masked) values can be a nuisance especially when dealing with
multimodel ensembles and having to compute multimodel statistics; different
numbers of missing data from dataset to dataset may introduce biases and
artifically assign more weight to the datasets that have less missing
data. This is handled in ESMValTool via the missing values masks: two types of
such masks are available, one for the multimodel case and another for the
single model case.

The multimodel missing values mask (``mask_fillvalues``) is a preprocessor step
that usually comes after all the single-model steps (regridding, area selection
etc) have been performed; in a nutshell, it combines missing values masks from
individual models into a multimodel missing values mask; the individual model
masks are built according to common criteria: the user chooses a time window in
which missing data points are counted, and if the number of missing data points
relative to the number of total data points in a window is less than a chosen
fractional theshold, the window is discarded i.e. all the points in the window
are masked (set to missing).

.. code-block:: yaml

    preprocessors:
      missing_values_preprocessor:
        mask_fillvalues:
          threshold_fraction: 0.95
          min_value: 19.0
          time_window: 10.0

In the example above, the fractional threshold for missing data vs. total data
is set to 95% and the time window is set to 10.0 (units of the time coordinate
units). Optionally, a minimum value threshold can be applied, in this case it
is set to 19.0 (in units of the variable units).

See also :func:`esmvalcore.preprocessor.mask_fillvalues`.

.. note::

   It is possible to use ``mask_fillvalues`` to create a combined multimodel
   mask (all the masks from all the analyzed models combined into a single
   mask); for that purpose setting the ``threshold_fraction`` to 0 will not
   discard any time windows, essentially keeping the original model masks and
   combining them into a single mask; here is an example:

   .. code-block:: yaml

       preprocessors:
         missing_values_preprocessor:
           mask_fillvalues:
             threshold_fraction: 0.0     # keep all missing values
             min_value: -1e20            # small enough not to alter the data
             #  time_window: 10.0        # this will not matter anymore

Minimum, maximum and interval masking
-------------------------------------

Thresholding on minimum and maximum accepted data values can also be performed:
masks are constructed based on the results of thresholding; inside and outside
interval thresholding and masking can also be performed. These functions are
``mask_above_threshold``, ``mask_below_threshold``, ``mask_inside_range``, and
``mask_outside_range``.

Thes functions always take a cube as first argument and either ``threshold``
for threshold masking or the pair ``minimum`, ``maximum`` for interval masking.

See also :func:`esmvalcore.preprocessor.mask_above_threshold` and related
functions.


.. _Horizontal regridding:

Horizontal regridding
=====================

Regridding is necessary when various datasets are available on a variety of
`lat-lon` grids and they need to be brought together on a common grid (for
various statistical operations e.g. multimodel statistics or for e.g. direct
inter-comparison or comparison with observational datasets). Regridding is
conceptually a very similar process to interpolation (in fact, the regridder
engine uses interpolation and extrapolation, with various schemes). The primary
difference is that interpolation is based on sample data points, while
regridding is based on the horizontal grid of another cube (the reference
grid).

The underlying regridding mechanism in ESMValTool uses the `cube.regrid()
<https://scitools.org.uk/iris/docs/latest/iris/iris/cube.html#iris.cube.Cube.regrid>`_
from Iris.

The use of the horizontal regridding functionality is flexible depending on
what type of reference grid and what interpolation scheme is preferred. Below
we show a few examples.

Regridding on a reference dataset grid
--------------------------------------

The example below shows how to regrid on the reference dataset ``ERA-Interim``
(observational data, but just as well CMIP, obs4mips, or ana4mips datasets can be used); in this case the `scheme` is `linear`.

.. code-block:: yaml

    preprocessors:
      regrid_preprocessor:
        regrid:
          target_grid: ERA-Interim
          scheme: linear

Regridding on an ``MxN`` grid specification
-------------------------------------------

The example below shows how to regrid on a reference grid with a cell
specification of ``2.5x2.5`` degrees. This is similar to regridding on
reference datasets, but in the previous case the reference dataset grid cell
specifications are not necessarily known a priori. Reegridding on an ``MxN``
cell specification is oftentimes used when operating on localized data.

.. code-block:: yaml

    preprocessors:
      regrid_preprocessor:
        regrid:
          target_grid: 2.5x2.5
          scheme: nearest

In this case the ``NearestNeighbour`` interpolation scheme is used (see below
for scheme definitions).

When using a ``MxN`` type of grid it is possible to offset the grid cell
centrepoints using the `lat_offset` and ``lon_offset`` arguments:

* ``lat_offset``: offsets the grid centers of the latitude coordinate w.r.t. the
  pole by half a grid step;
* ``lon_offset``: offsets the grid centers of the longitude coordinate
  w.r.t. Greenwich meridian by half a grid step.

.. code-block:: yaml

    preprocessors:
      regrid_preprocessor:
        regrid:
          target_grid: 2.5x2.5
          lon_offset: True
          lat_offset: True
          scheme: nearest

Regridding (interpolation, extrapolation) schemes
-------------------------------------------------

The schemes used for the interpolation and extrapolation operations needed by
the horizontal regridding functionality directly map to their corresponding
implementaions in Iris:

* ``linear``: `Linear(extrapolation_mode='mask') <https://scitools.org.uk/iris/docs/latest/iris/iris/analysis.html#iris.analysis.Linear>`_.
* ``linear_extrapolate``: `Linear(extrapolation_mode='extrapolate') <https://scitools.org.uk/iris/docs/latest/iris/iris/analysis.html#iris.analysis.Linear>`_.
* ``nearest``: `Nearest(extrapolation_mode='mask') <https://scitools.org.uk/iris/docs/latest/iris/iris/analysis.html#iris.analysis.Nearest>`_.
* ``area_weighted``: `AreaWeighted() <https://scitools.org.uk/iris/docs/latest/iris/iris/analysis.html#iris.analysis.AreaWeighted>`_.
* ``unstructured_nearest``: `UnstructuredNearest() <https://scitools.org.uk/iris/docs/latest/iris/iris/analysis.html#iris.analysis.UnstructuredNearest>`_.

See also :func:`esmvalcore.preprocessor.regrid`

.. note::

   For both vertical and horizontal regridding one can control the
   extrapolation mode when defining the interpolation scheme. Controlling the
   extrapolation mode allows us to avoid situations where extrapolating values
   makes little physical sense (e.g. extrapolating beyond the last data
   point). The extrapolation mode is controlled by the `extrapolation_mode`
   keyword. For the available interpolation schemes available in Iris, the
   extrapolation_mode keyword must be one of:

        * ``extrapolate`` – the extrapolation points will be calculated by
	  extending the gradient of the closest two points;
        * ``error`` – a ``ValueError`` exception will be raised, notifying an
	  attempt to extrapolate;
        * ``nan`` – the extrapolation points will be be set to NaN;
        * ``mask`` – the extrapolation points will always be masked, even if
	  the source data is not a ``MaskedArray``; or
        * ``nanmask`` – if the source data is a MaskedArray the extrapolation
	  points will be masked, otherwise they will be set to NaN.

.. note::

   The regridding mechanism is (at the moment) done with fully realized data in
   memory, so depending on how fine the target grid is, it may use a rather
   large amount of memory. Empirically target grids of up to ``0.5x0.5``
   degrees should not produce any memory-related issues, but be advised that
   for resolutions of ``< 0.5`` degrees the regridding becomes very slow and
   will use a lot of memory.


.. _multi-model statistics:

Multi-model statistics
======================
Computing multi-model statistics is an integral part of model analysis and
evaluation: individual models display a variety of biases depending on model
set-up, initial conditions, forcings and implementation; comparing model data
to observational data, these biases have a significanly lower statistical
impact when using a multi-model ensemble. ESMValTool has the capability of
computing a number of multi-model statistical measures: using the preprocessor
module ``multi_model_statistics`` will enable the user to ask for either a
multi-model ``mean`` and/or ``median`` with a set of argument parameters passed
to ``multi_model_statistics``.

Multimodel statistics in ESMValTool are computed along the time axis, and as
such, can be computed across a common overlap in time (by specifying ``span:
overlap`` argument) or across the full length in time of each model (by
specifying ``span: full`` argument).

Restrictive computation is also available by excluding  any set of models that
the user will not want to include in the statistics (by setting ``exclude:
[excluded models list]`` argument). The implementation has a few restrictions
that apply to the input data: model datasets must have consistent shapes, and
from a statistical point of view, this is needed since weights are not yet
implemented; also higher dimensional data is not supported (i.e. anything with
dimensionality higher than four: time, vertical axis, two horizontal axes).

.. code-block:: yaml

    preprocessors:
      multimodel_preprocessor:
        multi_model_statistics:
          span: overlap
          statistics: [mean, median]
          exclude: [NCEP]

see also :func:`esmvalcore.preprocessor.multi_model_statistics`.

.. note::

   Note that the multimodel array operations, albeit performed in
   per-time/per-horizontal level loops to save memory, could, however, be
   rather memory-intensive (since they are not performed lazily as
   yet). The Section on :ref:`Memory use` details the memory intake
   for different run scenarios, but as a thumb rule, for the multimodel
   preprocessor, the expected maximum memory intake could be approximated as
   the number of datasets multiplied by the average size in memory for one
   dataset.

.. _time operations:

Time manipulation
=================
The ``_time.py`` module contains the following preprocessor functions:

* extract_time_: Extract a time range from a cube.
* extract_season_: Extract only the times that occur within a specific season.
* extract_month_: Extract only the times that occur within a specific month.
* daily_statistics_: Compute statistics for each day
* monthly_statistics_: Compute statistics for each month
* seasonal_statistics_: Compute statistics for each season
* annual_statistics_: Compute statistics for each year
* decadal_statistics_: Compute statistics for each decade
* climate_statistics_: Compute statistics for the full period
* anomalies_: Compute anomalies
* regrid_time_: Aligns the time axis of each dataset to have common time
  points and calendars.

Statistics functions are applied by default in the order they appear in the
list. For example, the following example applied to hourly data will retrieve
the minimum values for the full period (by season) of the monthly mean of the
daily maximum of any given variable.

.. code-block:: yaml

    daily_statistics:
      operator: max

    monthly_statistics:
      operator: mean

    climate_statistics:
      operator: min
      period: season


.. _extract_time:

``extract_time``
----------------

This function subsets a dataset between two points in times. It removes all
times in the dataset before the first time and after the last time point.
The required arguments are relatively self explanatory:

* ``start_year``
* ``start_month``
* ``start_day``
* ``end_year``
* ``end_month``
* ``end_day``

These start and end points are set using the datasets native calendar.
All six arguments should be given as integers - the named month string
will not be accepted.

See also :func:`esmvalcore.preprocessor.extract_time`.

.. _extract_season:

``extract_season``
------------------

Extract only the times that occur within a specific season.

This function only has one argument: ``season``. This is the named season to
extract. ie: DJF, MAM, JJA, SON.

Note that this function does not change the time resolution. If your original
data is in monthly time resolution, then this function will return three
monthly datapoints per year.

If you want the seasonal average, then this function needs to be combined with
the seasonal_mean function, below.

See also :func:`esmvalcore.preprocessor.extract_season`.

.. _extract_month:

``extract_month``
-----------------

The function extracts the times that occur within a specific month.
This function only has one argument: ``month``. This value should be an integer
between 1 and 12 as the named month string will not be accepted.

See also :func:`esmvalcore.preprocessor.extract_month`.

.. _daily_statistics:

``daily_statistics``
--------------------

This function produces statistics for each day in the dataset.

Parameters:
    * operator: operation to apply. Accepted values are 'mean',
      'median', 'std_dev', 'min', 'max' and 'sum'. Default is 'mean'

See also :func:`esmvalcore.preprocessor.daily_statistics`.

.. _monthly_statistics:

``monthly_statistics``
----------------------

This function produces statistics for each month in the dataset.

Parameters:
    * operator: operation to apply. Accepted values are 'mean',
      'median', 'std_dev', 'min', 'max' and 'sum'. Default is 'mean'

See also :func:`esmvalcore.preprocessor.monthly_statistics`.

.. _seasonal_statistics:

``seasonal_statistics``
-----------------------

This function produces statistics for each season (DJF, MAM, JJA, SON) in the
dataset. Note that this function will not check for missing time points.
For instance, if you are looking at the DJF field, but your datasets
starts on January 1st, the first DJF field will only contain data
from January and February.

We recommend using the extract_time to start the dataset from the following
December and remove such biased initial datapoints.

Parameters:
    * operator: operation to apply. Accepted values are 'mean',
      'median', 'std_dev', 'min', 'max' and 'sum'. Default is 'mean'

See also :func:`esmvalcore.preprocessor.seasonal_mean`.

.. _annual_statistics:

``annual_statistics``
---------------------

This function produces statistics for each year.

Parameters:
    * operator: operation to apply. Accepted values are 'mean',
      'median', 'std_dev', 'min', 'max' and 'sum'. Default is 'mean'

See also :func:`esmvalcore.preprocessor.annual_statistics`.

.. _decadal_statistics:

``decadal_statistics``
----------------------

This function produces statistics for each decade.

Parameters:
    * operator: operation to apply. Accepted values are 'mean',
      'median', 'std_dev', 'min', 'max' and 'sum'. Default is 'mean'

See also :func:`esmvalcore.preprocessor.decadal_statistics`.

.. _climate_statistics:

``climate_statistics``
----------------------

This function produces statistics for the whole dataset. It can produce scalars
(if the full period is chosen) or daily, monthly or seasonal statics.

Parameters:
    * operator: operation to apply. Accepted values are 'mean', 'median',
      'std_dev', 'min', 'max' and 'sum'. Default is 'mean'

    * period: define the granularity of the statistics: get values for the
      full period, for each month or day of year.
      Available periods: 'full', 'season', 'seasonal', 'monthly', 'month',
      'mon', 'daily', 'day'. Default is 'full'

Examples:
    * Monthly climatology:

        .. code-block:: yaml

            climate_statistics:
                operator: mean
                period: month

    * Daily maximum for the full period:

        .. code-block:: yaml

            climate_statistics:
                operator: max
                period: day

    * Minimum value in the period:

        .. code-block:: yaml

            climate_statistics:
                operator: min
                period: full

See also :func:`esmvalcore.preprocessor.climate_statistics`.

.. _anomalies:

``anomalies``
----------------------

This function computes the anomalies for the whole dataset. It can compute
anomalies from the full, seasonal, monthly and daily climatologies.

Parameters:
    * period: define the granularity of the climatology to use:
      full period, seasonal, monthly or daily.
      Available periods: 'full', 'season', 'seasonal', 'monthly', 'month',
      'mon', 'daily', 'day'. Default is 'full'

Examples:
    * Anomalies from the monthly climatology:

        .. code-block:: yaml

            anomalies:
                period: month

    * Anomalies from the full period climatology:

        .. code-block:: yaml

            anomalies:

See also :func:`esmvalcore.preprocessor.anomalies`.


.. _regrid_time:

``regrid_time``
---------------

This function aligns the time points of each component dataset so that the Iris
cubes from different datasets can be subtracted. The operation makes the
datasets time points common and sets common calendars; it also resets the time
bounds and auxiliary coordinates to reflect the artifically shifted time
points. Current implementation for monthly and daily data; the ``frequency`` is
set automatically from the variable CMOR table unless a custom ``frequency`` is
set manually by the user in recipe.

See also :func:`esmvalcore.preprocessor.regrid_time`.


.. _area operations:

Area manipulation
=================
The area manipulation module contains the following preprocessor functions:

* extract_region_: Extract a region from a cube based on ``lat/lon``
  corners.
* extract_named_regions_: Extract a specific region from in the region
  cooordinate.
* extract_shape_: Extract a region defined by a shapefile.
* zonal_means_: Calculates the zonal or meridional means.
* area_statistics_: Calculates the average value over a region.


``extract_region``
------------------

This function masks data outside a rectagular region requested. The boundairies
of the region are provided as latitude and longitude coordinates in the
arguments:

* ``start_longitude``
* ``end_longitude``
* ``start_latitude``
* ``end_latitude``

Note that this function can only be used to extract a rectangular region. Use
``extract_shape`` to extract any other shaped region from a shapefile.

See also :func:`esmvalcore.preprocessor.extract_region`.


``extract_named_regions``
-------------------------

This function extracts a specific named region from the data. This function
takes the following argument: ``regions`` which is either a string or a list
of strings of named regions. Note that the dataset must have a ``region``
cooordinate which includes a list of strings as values. This function then
matches the named regions against the requested string.

See also :func:`esmvalcore.preprocessor.extract_named_regions`.


``extract_shape``
-------------------------

Extract a shape or a representative point for this shape from
the data.

Parameters:
	* ``shapefile``: path to the shapefile containing the geometry of the region to be
	  extracted. This path can be relative to ``auxiliary_data_dir`` defined in
	  the :ref:`user configuration file`.
	* ``method``: the method to select the region, selecting either all points
	  contained by the shape or a single representative point. Choose either
	  'contains' or 'representative'. If not a single grid point is contained
	  in the shape, a representative point will be selected.
	* ``crop``: by default extract_region_ will be used to crop the data to a
	  minimal rectangular region containing the shape. Set to ``false`` to only
	  mask data outside the shape. Data on irregular grids will not be cropped.

Examples:
    * Extract the shape of the river Elbe from a shapefile:

        .. code-block:: yaml

            extract_shape:
              shapefile: Elbe.shp
              method: contains

See also :func:`esmvalcore.preprocessor.extract_shape`.


``zonal_means``
---------------

The function calculates the zonal or meridional means. While this function is
named ``zonal_mean``, it can be used to apply several different operations in
an zonal or meridional direction. This function takes two arguments:

* ``coordinate``: Which direction to apply the operation: latitude or longitude
* ``mean_type``: Which operation to apply: mean, std_dev, variance, median,
  min, max or sum

See also :func:`esmvalcore.preprocessor.zonal_means`.


``area_statistics``
-------------------

This function calculates the average value over a region - weighted by the cell
areas of the region. This function takes the argument, ``operator``: the name
of the operation to apply.

This function can be used to apply several different operations in the
horizonal plane: mean, standard deviation, median variance, minimum and maximum.

Note that this function is applied over the entire dataset. If only a specific
region, depth layer or time period is required, then those regions need to be
removed using other preprocessor operations in advance.

See also :func:`esmvalcore.preprocessor.area_statistics`.


.. _volume operations:

Volume manipulation
===================
The ``_volume.py`` module contains the following preprocessor functions:

* ``extract_volume``: Extract a specific depth range from a cube.
* ``volume_statistics``: Calculate the volume-weighted average.
* ``depth_integration``: Integrate over the depth dimension.
* ``extract_transect``: Extract data along a line of constant latitude or
  longitude.
* ``extract_trajectory``: Extract data along a specified trajectory.


``extract_volume``
------------------

Extract a specific range in the `z`-direction from a cube.  This function
takes two arguments, a minimum and a maximum (``z_min`` and ``z_max``,
respectively) in the `z`-direction.

Note that this requires the requested `z`-coordinate range to be the same sign
as the Iris cube. ie, if the cube has `z`-coordinate as negative, then
``z_min`` and ``z_max`` need to be negative numbers.

See also :func:`esmvalcore.preprocessor.extract_volume`.


``volume_statistics``
---------------------

This function calculates the volume-weighted average across three dimensions,
but maintains the time dimension.

This function takes the argument: ``operator``, which defines the operation to
apply over the volume.

No depth coordinate is required as this is determined by Iris. This function
works best when the ``fx_files`` provide the cell volume.

See also :func:`esmvalcore.preprocessor.volume_statistics`.


``depth_integration``
---------------------

This function integrates over the depth dimension. This function does a
weighted sum along the `z`-coordinate, and removes the `z` direction of the
output cube. This preprocessor takes no arguments.

See also :func:`esmvalcore.preprocessor.depth_integration`.


``extract_transect``
--------------------

This function extracts data along a line of constant latitude or longitude.
This function takes two arguments, although only one is strictly required.
The two arguments are ``latitude`` and ``longitude``. One of these arguments
needs to be set to a float, and the other can then be either ignored or set to
a minimum or maximum value.

For example, if we set latitude to 0 N and leave longitude blank, it would
produce a cube along the Equator. On the other hand, if we set latitude to 0
and then set longitude to ``[40., 100.]`` this will produce a transect of the
Equator in the Indian Ocean.

See also :func:`esmvalcore.preprocessor.extract_transect`.


``extract_trajectory``
----------------------

This function extract data along a specified trajectory.
The three areguments are: ``latitudes``, ``longitudes`` and number of point
needed for extrapolation ``number_points``.

If two points are provided, the ``number_points`` argument is used to set a
the number of places to extract between the two end points.

If more than two points are provided, then ``extract_trajectory`` will produce
a cube which has extrapolated the data of the cube to those points, and
``number_points`` is not needed.

Note that this function uses the expensive ``interpolate`` method from
``Iris.analysis.trajectory``, but it may be neccesary for irregular grids.

See also :func:`esmvalcore.preprocessor.extract_trajectory`.

.. _detrend:

Detrend
=======

ESMValTool also supports detrending along any dimension using
the preprocessor function 'detrend'.
This function has two parameters:

* ``dimension``: dimension to apply detrend on. Default: "time"
* ``method``: It can be ``linear`` or ``constant``. Default: ``linear``

If method is ``linear``, detrend will calculate the linear trend along the
selected axis and substract it to the data. For example, this can be used to
remove the linear trend caused by climate change on some variables is selected
dimension is time.

If method is ``constant``, detrend will compute the mean along that dimension
and substract it from the data

See also :func:`esmvalcore.preprocessor.detrend`.

.. _unit conversion:

Unit conversion
===============

Converting units is also supported. This is particularly useful in
cases where different datasets might have different units, for example
when comparing CMIP5 and CMIP6 variables where the units have changed
or in case of observational datasets that are delivered in different
units.

In these cases, having a unit conversion at the end of the processing
will guarantee homogeneous input for the diagnostics.

.. note::
   Conversion is only supported between compatible units! In other
   words, converting temperature units from ``degC`` to ``Kelvin`` works
   fine, changing precipitation units from a rate based unit to an
   amount based unit is not supported at the moment.

See also :func:`esmvalcore.preprocessor.convert_units`.


.. _Memory use:

Information on maximum memory required
======================================
In the most general case, we can set upper limits on the maximum memory the
anlysis will require:


``Ms = (R + N) x F_eff - F_eff`` - when no multimodel analysis is performed;

``Mm = (2R + N) x F_eff - 2F_eff`` - when multimodel analysis is performed;

where

* ``Ms``: maximum memory for non-multimodel module
* ``Mm``: maximum memory for multimodel module
* ``R``: computational efficiency of module; `R` is typically 2-3
* ``N``: number of datasets
* ``F_eff``: average size of data per dataset where ``F_eff = e x f x F``
  where ``e`` is the factor that describes how lazy the data is (``e = 1`` for
  fully realized data) and ``f`` describes how much the data was shrunk by the
  immediately previous module, e.g. time extraction, area selection or level
  extraction; note that for fix_data ``f`` relates only to the time extraction,
  if data is exact in time (no time selection) ``f = 1`` for fix_data so for
  cases when we deal with a lot of datasets ``R + N \approx N``, data is fully
  realized, assuming an average size of 1.5GB for 10 years of `3D` netCDF data,
  ``N`` datasets will require:


``Ms = 1.5 x (N - 1)`` GB

``Mm = 1.5 x (N - 2)`` GB

As a rule of thumb, the maximum required memory at a certain time for
multimodel analysis could be estimated by multiplying the number of datasets by
the average file size of all the datasets; this memory intake is high but also
assumes that all data is fully realized in memory; this aspect will gradually
change and the amount of realized data will decrease with the increase of
``dask`` use.
