.. _preprocessor:

************
Preprocessor
************

In this section, each of the preprocessor modules is described,
roughly following the default order in which preprocessor functions are applied:

* :ref:`Overview`
* :ref:`stat_preprocs`
* :ref:`Variable derivation`
* :ref:`CMOR check and dataset-specific fixes`
* :ref:`preprocessors_using_supplementary_variables`
* :ref:`Vertical interpolation`
* :ref:`Weighting`
* :ref:`Land/Sea/Ice masking`
* :ref:`Horizontal regridding`
* :ref:`Masking of missing values`
* :ref:`Ensemble statistics`
* :ref:`Multi-model statistics`
* :ref:`Time operations`
* :ref:`Area operations`
* :ref:`Volume operations`
* :ref:`Cycles`
* :ref:`Trend`
* :ref:`Detrend`
* :ref:`Rolling window statistics`
* :ref:`Unit conversion`
* :ref:`comparison_with_ref`
* :ref:`Other`

See :ref:`preprocessor_functions` for implementation details and the exact default order.

.. _overview:

Overview
========

The ESMValCore preprocessor can be used to perform a broad range of operations
on the input data before diagnostics or metrics are applied. The preprocessor
performs these operations in a centralized, documented and efficient way, thus
reducing the data processing load on the diagnostics side.  For an overview of
the preprocessor structure see the :ref:`Preprocessors`.

Each of the preprocessor operations is written in a dedicated python module and
all of them receive and return an instance of
:obj:`iris.cube.Cube`, working
sequentially on the data with no interactions between them. The order in which
the preprocessor operations is applied is set by default to minimize
the loss of information due to, for example, temporal and spatial subsetting or
multi-model averaging. Nevertheless, the user is free to change such order to
address specific scientific requirements, but keeping in mind that some
operations must be necessarily performed in a specific order. This is the case,
for instance, for multi-model statistics, which required the model to be on a
common grid and therefore has to be called after the regridding module.


.. _stat_preprocs:

Statistical preprocessors
=========================

Many preprocessors calculate statistics over data.
Those preprocessors typically end with ``_statistics``, e.g.,
:func:`~esmvalcore.preprocessor.area_statistics` or
:func:`~esmvalcore.preprocessor.multi_model_statistics`.
All these preprocessors support the options `operator`, which directly
correspond to :class:`iris.analysis.Aggregator` objects used to perform the
statistical calculations.
In addition, arbitrary keyword arguments can be passed which are directly
passed to the corresponding :class:`iris.analysis.Aggregator` object.

.. note::
    The preprocessors :func:`~esmvalcore.preprocessor.multi_model_statistics`
    and :func:`~esmvalcore.preprocessor.ensemble_statistics` support the
    computation of multiple statistics at the same time.
    In these cases, they are defined by the option `statistics` (instead of
    `operator`), which takes a list of possible operators.
    Each operator can be given as single string or as dictionary.
    In the latter case, the dictionary needs the keyword `operator`
    (corresponding to the `operator` as above).
    All other keywords are interpreted as keyword arguments for the given
    operator.

Some operators support weights for some preprocessors (see following table),
which are used by default.
The following operators are currently fully supported; other operators might be
supported too if proper keyword arguments are specified:

.. _supported_stat_operator:

============================== ================================================= =====================================
`operator`                     Corresponding :class:`~iris.analysis.Aggregator`  Weighted? [#f1]_
============================== ================================================= =====================================
``gmean``                      :const:`iris.analysis.GMEAN`                      no
``hmean``                      :const:`iris.analysis.HMEAN`                      no
``max``                        :const:`iris.analysis.MAX`                        no
``mean``                       :const:`iris.analysis.MEAN`                       yes
``median``                     :const:`iris.analysis.MEDIAN` [#f2]_                no
``min``                        :const:`iris.analysis.MIN`                        no
``peak``                       :const:`iris.analysis.PEAK`                       no
``percentile``                 :const:`iris.analysis.PERCENTILE`                 no
``rms``                        :const:`iris.analysis.RMS`                        yes
``std_dev``                    :const:`iris.analysis.STD_DEV`                    no
``sum``                        :const:`iris.analysis.SUM`                        yes
``variance``                   :const:`iris.analysis.VARIANCE`                   no
``wpercentile``                :const:`iris.analysis.WPERCENTILE`                yes
============================== ================================================= =====================================

.. [#f1] The following preprocessor support weighted statistics by default:
    :func:`~esmvalcore.preprocessor.area_statistics`: weighted by grid cell
    areas (see also :ref:`preprocessors_using_supplementary_variables`);
    :func:`~esmvalcore.preprocessor.climate_statistics`: weighted by lengths of
    time intervals; :func:`~esmvalcore.preprocessor.volume_statistics`:
    weighted by grid cell volumes (see also
    :ref:`preprocessors_using_supplementary_variables`);
    :func:`~esmvalcore.preprocessor.axis_statistics`: weighted by
    corresponding coordinate bounds.
.. [#f2] :const:`iris.analysis.MEDIAN` is not lazy, but much faster than
    :const:`iris.analysis.PERCENTILE`. For a lazy median, use ``percentile``
    with the keyword argument ``percent: 50``.

Examples
--------

Calculate the global (weighted) mean:

.. code-block:: yaml

  preprocessors:
    global_mean:
      area_statistics:
        operator: mean

Calculate zonal maximum.

.. code-block:: yaml

  preprocessors:
    zonal_max:
      zonal_statistics:
        operator: max

Calculate the 95% percentile over each month separately (will result in 12 time
steps, one for January, one for February, etc.):

.. code-block:: yaml

  preprocessors:
    monthly_percentiles:
      climate_statistics:
        period: monthly
        operator: percentile
        percent: 95.0

Calculate multi-model median, 5%, and 95% percentiles:

.. code-block:: yaml

  preprocessors:
    mm_stats:
      multi_model_statistics:
        span: overlap
        statistics:
          - operator: percentile
            percent: 5
          - operator: median
          - operator: percentile
            percent: 95

Calculate the global non-weighted root mean square:

.. code-block:: yaml

  preprocessors:
    global_mean:
      area_statistics:
        operator: rms
        weights: false

.. warning::

  The disabling of weights by specifying the keyword argument ``weights:
  false`` needs to be used with great care; from a scientific standpoint, we
  strongly recommend to **not** use it!


.. _Variable derivation:

Variable derivation
===================
The variable derivation module allows to derive variables which are not in the
CMIP standard data request using standard variables as input. The typical use
case of this operation is the evaluation of a variable which is only available
in an observational dataset but not in the models. In this case a derivation
function is provided by the ESMValCore in order to calculate the variable and
perform the comparison. For example, several observational datasets deliver
total column ozone as observed variable (`toz`), but CMIP models only provide
the ozone 3D field. In this case, a derivation function is provided to
vertically integrate the ozone and obtain total column ozone for direct
comparison with the observations.

The tool will also look in other ``mip`` tables for the same ``project`` to find
the definition of derived variables. To contribute a completely new derived
variable, it is necessary to define a name for it and to provide the
corresponding CMOR table. This is to guarantee the proper metadata definition
is attached to the derived data. Such custom CMOR tables are collected as part
of the `ESMValCore package <https://github.com/ESMValGroup/ESMValCore/tree/main/esmvalcore/cmor/tables/custom>`_.
By default, the variable derivation will be applied only if the variable is not
already available in the input data, but the derivation can be forced by
setting the ``force_derivation`` flag.

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

Data preprocessed by ESMValCore is automatically checked against its
CMOR definition. To reduce the impact of this check while maintaining
it as reliable as possible, it is split in two parts: one will check
the metadata and will be done just after loading and concatenating the
data and the other one will check the data itself and will be applied
after all extracting operations are applied to reduce the amount of
data to process.

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
    - Minute differences between the required and actual vertical coordinate values


Dataset specific fixes
----------------------

Sometimes, the checker will detect errors that it can not fix by itself.
ESMValCore deals with those issues by applying specific fixes for those
datasets that require them. Fixes are applied at three different preprocessor
steps:

    - ``fix_file``: apply fixes directly to a copy of the file.
      Copying the files is costly, so only errors that prevent Iris to load the
      file are fixed here.
      See :func:`esmvalcore.preprocessor.fix_file`.

    - ``fix_metadata``: metadata fixes are done just before concatenating the
      cubes loaded from different files in the final one.
      Automatic metadata fixes are also applied at this step.
      See :func:`esmvalcore.preprocessor.fix_metadata`.

    - ``fix_data``: data fixes are applied before starting any operation that
      will alter the data itself.
      Automatic data fixes are also applied at this step.
      See :func:`esmvalcore.preprocessor.fix_data`.

To get an overview on data fixes and how to implement new ones, please go to
:ref:`fixing_data`.

.. _preprocessors_using_supplementary_variables:

Supplementary variables (ancillary variables and cell measures)
===============================================================
The following preprocessor functions either require or prefer using an
`ancillary variable <https://cfconventions.org/cf-conventions/cf-conventions.html#ancillary-data>`_
or
`cell measure <https://cfconventions.org/cf-conventions/cf-conventions.html#cell-measures>`_
to perform their computations.
In ESMValCore we call both types of variables "supplementary variables".

===================================================================== ============================== =====================================
Preprocessor                                                          Variable short name            Variable standard name
===================================================================== ============================== =====================================
:ref:`area_statistics<area_statistics>` [#f4]_                        ``areacella``, ``areacello``   cell_area
:ref:`mask_landsea<land/sea/ice masking>` [#f4]_                      ``sftlf``, ``sftof``           land_area_fraction, sea_area_fraction
:ref:`mask_landseaice<ice masking>` [#f3]_                            ``sftgif``                     land_ice_area_fraction
:ref:`volume_statistics<volume_statistics>` [#f4]_                    ``volcello``, ``areacello``    ocean_volume, cell_area
:ref:`weighting_landsea_fraction<land/sea fraction weighting>` [#f3]_ ``sftlf``, ``sftof``           land_area_fraction, sea_area_fraction
:ref:`distance_metric<distance_metric>` [#f5]_                        ``areacella``, ``areacello``   cell_area
:ref:`histogram<histogram>` [#f5]_                                    ``areacella``, ``areacello``   cell_area
===================================================================== ============================== =====================================

.. [#f3] This preprocessor requires at least one of the mentioned supplementary
    variables. If none is defined in the recipe, automatically look for them.
    If none is found, an error will be raised.
.. [#f4] This preprocessor prefers at least one of the mentioned supplementary
    variables. If none is defined in the recipe, automatically look for them.
    If none is found, a warning will be raised (but no error).
.. [#f5] This preprocessor optionally takes one of the mentioned supplementary
    variables. If none is defined in the recipe, none is added.

Only one of the listed variables is required. Supplementary variables can be
defined in the recipe as described in :ref:`supplementary_variables`.
If the automatic selection does not give the desired result, specify the
supplementary variables in the recipe as described in
:ref:`supplementary_variables`.

By default, supplementary variables will be removed from the
variable before saving it to file because they can be as big as the main
variable.
To keep the supplementary variables, disable the preprocessor
function :func:`esmvalcore.preprocessor.remove_supplementary_variables` that
removes them by setting ``remove_supplementary_variables: false`` in the
preprocessor in the recipe.

Examples
--------

Compute the global mean surface air temperature, while
:ref:`automatically selecting the best matching supplementary dataset <supplementary_dataset_wildcards>`:

.. code-block:: yaml

  datasets:
    - dataset: BCC-ESM1
      project: CMIP6
      ensemble: r1i1p1f1
      grid: gn
    - dataset: MPI-ESM-MR
      project: CMIP5
      ensemble: r1i1p1,

  preprocessors:
    global_mean:
      area_statistics:
        operator: mean

  diagnostics:
    example_diagnostic:
      description: Global mean temperature.
      variables:
        tas:
          mip: Amon
          preprocessor: global_mean
          exp: historical
          timerange: '1990/2000'
          supplementary_variables:
            - short_name: areacella
              mip: fx
              exp: '*'
              activity: '*'
              ensemble: '*'
      scripts: null

Attach the land area fraction as an ancillary variable to surface air
temperature and store both in the same file:

.. code-block:: yaml

  datasets:
    - dataset: BCC-ESM1
      ensemble: r1i1p1f1
      grid: gn

  preprocessors:
    keep_land_area_fraction:
      remove_supplementary_variables: false

  diagnostics:
    example_diagnostic:
      description: Attach land area fraction.
      variables:
        tas:
          mip: Amon
          project: CMIP6
          preprocessor: keep_land_area_fraction
          exp: historical
          timerange: '1990/2000'
          supplementary_variables:
            - short_name: sftlf
              mip: fx
              exp: 1pctCO2
      scripts: null


Automatically define the required ancillary variable (``sftlf`` in this case)
and cell measure (``areacella``), but do not use ``areacella`` for dataset
``BCC-ESM1``:

.. code-block:: yaml

  datasets:
    - dataset: BCC-ESM1
      project: CMIP6
      ensemble: r1i1p1f1
      grid: gn
      supplementary_variables:
        - short_name: areacella
          skip: true
    - dataset: MPI-ESM-MR
      project: CMIP5
      ensemble: r1i1p1

  preprocessors:
    global_land_mean:
      mask_landsea:
        mask_out: sea
      area_statistics:
        operator: mean

  diagnostics:
    example_diagnostic:
      description: Global mean temperature.
      variables:
        tas:
          mip: Amon
          preprocessor: global_land_mean
          exp: historical
          timerange: '1990/2000'
      scripts: null

.. _Vertical interpolation:

Vertical interpolation
======================
Vertical level selection is an important aspect of data preprocessing since it
allows the scientist to perform a number of metrics specific to certain levels
(whether it be air pressure or depth, e.g. the Quasi-Biennial-Oscillation (QBO)
u30 is computed at 30 hPa). Dataset native vertical grids may not come with the
desired set of levels, so an interpolation operation will be needed to regrid
the data vertically. ESMValCore can perform this vertical interpolation via the
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
          scheme: linear_extrapolate

By default, vertical interpolation is performed in the dimension coordinate of
the z axis. If you want to explicitly declare the z axis coordinate to use
(for example, ``air_pressure``' in variables that are provided in model levels
and not pressure levels) you can override that automatic choice by providing
the name of the desired coordinate:

.. code-block:: yaml

    preprocessors:
      preproc_select_levels_from_dataset:
        extract_levels:
          levels: ERA-Interim
          scheme: linear_extrapolate
          coordinate: air_pressure

If ``coordinate`` is specified, pressure levels (if present) can be converted
to height levels and vice versa using the US standard atmosphere. E.g.
``coordinate = altitude`` will convert existing pressure levels
(air_pressure) to height levels (altitude);
``coordinate = air_pressure`` will convert existing height levels
(altitude) to pressure levels (air_pressure).

If the requested levels are very close to the values in the input data,
the function will just select the available levels instead of interpolating.
The meaning of 'very close' can be changed by providing the parameters:

* ``rtol``
    Relative tolerance for comparing the levels in the input data to the requested
    levels. If the levels are sufficiently close, the requested levels
    will be assigned to the vertical coordinate and no interpolation will take place.
    The default value is 10^-7.
* ``atol``
    Absolute tolerance for comparing the levels in the input data to the requested
    levels. If the levels are sufficiently close, the requested levels
    will be assigned to the vertical coordinate and no interpolation will take place.
    By default, `atol` will be set to 10^-7 times the mean value of
    of the available levels.

.. _Vertical interpolation schemes:

Schemes for vertical interpolation and extrapolation
----------------------------------------------------

The vertical interpolation currently supports the following schemes:

* ``linear``: Linear interpolation without extrapolation, i.e., extrapolation
  points will be masked even if the source data is not a masked array.
* ``linear_extrapolate``: Linear interpolation with **nearest-neighbour**
  extrapolation, i.e., extrapolation points will take their value from the
  nearest source point.
* ``nearest``: Nearest-neighbour interpolation without extrapolation, i.e.,
  extrapolation points will be masked even if the source data is not a masked
  array.
* ``nearest_extrapolate``: Nearest-neighbour interpolation with nearest-neighbour
  extrapolation, i.e., extrapolation points will take their value from the
  nearest source point.
* See also :func:`esmvalcore.preprocessor.extract_levels`.
* See also :func:`esmvalcore.preprocessor.get_cmor_levels`.

.. note::

   Controlling the extrapolation mode allows us to avoid situations where
   extrapolating values makes little physical sense (e.g. extrapolating beyond
   the last data point).


.. _weighting:

Weighting
=========

.. _land/sea fraction weighting:

Land/sea fraction weighting
---------------------------

This preprocessor allows weighting of data by land or sea fractions. In other
words, this function multiplies the given input field by a fraction in the range 0-1 to
account for the fact that not all grid points are completely land- or sea-covered.

The application of this preprocessor is very important for most carbon cycle variables (and
other land surface outputs), which are e.g. reported in units of
:math:`kgC~m^{-2}`. Here, the surface unit actually refers to 'square meter of land/sea' and
NOT 'square meter of gridbox'. In order to integrate these globally or
regionally one has to weight by both the surface quantity and the
land/sea fraction.

For example, to weight an input field with the land fraction, the following
preprocessor can be used:

.. code-block:: yaml

    preprocessors:
      preproc_weighting:
        weighting_landsea_fraction:
          area_type: land
          exclude: ['CanESM2', 'reference_dataset']

Allowed arguments for the keyword ``area_type`` are ``land`` (fraction is 1
for grid cells with only land surface, 0 for grid cells with only sea surface
and values in between 0 and 1 for coastal regions) and ``sea`` (1 for
sea, 0 for land, in between for coastal regions). The optional argument
``exclude`` allows to exclude specific datasets from this preprocessor, which
is for example useful for climate models which do not offer land/sea fraction
files. This arguments also accepts the special dataset specifiers
``reference_dataset`` and ``alternative_dataset``.

This function requires a land or sea area fraction `ancillary variable`_.
This supplementary variable, either ``sftlf`` or ``sftof``, should be attached
to the main dataset as described in :ref:`supplementary_variables`.

See also :func:`esmvalcore.preprocessor.weighting_landsea_fraction`.


.. _masking:

Masking
=======

Introduction to masking
-----------------------

Certain metrics and diagnostics need to be computed and performed on specific
domains on the globe. The preprocessor supports filtering
the input data on continents, oceans/seas and ice. This is achieved by masking
the model data and keeping only the values associated with grid points that
correspond to, e.g., land, ocean or ice surfaces, as specified by the
user. Where possible, the masking is realized using the standard mask files
provided together with the model data as part of the CMIP data request (the
so-called ancillary variable). In the absence of these files, the Natural Earth masks
are used: although these are not model-specific, they represent a good
approximation since they have a much higher resolution than most of the models
and they are regularly updated with changing geographical features.

.. _land/sea/ice masking:

Land-sea masking
----------------

To mask out a certain domain (e.g., sea) in the preprocessor,
``mask_landsea`` can be used:

.. code-block:: yaml

    preprocessors:
      preproc_mask:
        mask_landsea:
          mask_out: sea

and requires only one argument: ``mask_out``: either ``land`` or ``sea``.

This function prefers using a land or sea area fraction `ancillary variable`_,
but if it is not available it will compute a mask based on
`Natural Earth <https://www.naturalearthdata.com>`_ shapefiles.
This supplementary variable, either ``sftlf`` or ``sftof``, can be attached
to the main dataset as described in :ref:`supplementary_variables`.

If the corresponding ancillary variable is not available (which is
the case for some models and almost all observational datasets), the
preprocessor attempts to mask the data using Natural Earth mask files (that are
vectorized rasters). As mentioned above, the spatial resolution of the the
Natural Earth masks are much higher than any typical global model (10m for
land and glaciated areas and 50m for ocean masks).

See also :func:`esmvalcore.preprocessor.mask_landsea`.

.. _ice masking:

Ice masking
-----------

For masking out ice sheets, the preprocessor uses a different
function, to ensure that both land and sea or ice can be masked out without
losing generality. To mask ice out, ``mask_landseaice`` can be used:

.. code-block:: yaml

  preprocessors:
    preproc_mask:
      mask_landseaice:
        mask_out: ice

and requires only one argument: ``mask_out``: either ``landsea`` or ``ice``.

This function requires a land ice area fraction `ancillary variable`_.
This supplementary variable ``sftgif`` should be attached to the main dataset as
described in :ref:`supplementary_variables`.

See also :func:`esmvalcore.preprocessor.mask_landseaice`.

Glaciated masking
-----------------

For masking out glaciated areas a Natural Earth shapefile is used. To mask
glaciated areas out, ``mask_glaciated`` can be used:

.. code-block:: yaml

  preprocessors:
    preproc_mask:
      mask_glaciated:
        mask_out: glaciated

and it requires only one argument: ``mask_out``: only ``glaciated``.

See also :func:`esmvalcore.preprocessor.mask_landseaice`.

.. _masking of missing values:

Missing values masks
--------------------

Missing (masked) values can be a nuisance especially when dealing with
multi-model ensembles and having to compute multi-model statistics; different
numbers of missing data from dataset to dataset may introduce biases and
artificially assign more weight to the datasets that have less missing data.
This is handled via the missing values masks: two types of such
masks are available, one for the multi-model case and another for the single
model case.

The multi-model missing values mask (``mask_fillvalues``) is a preprocessor step
that usually comes after all the single-model steps (regridding, area selection
etc) have been performed; in a nutshell, it combines missing values masks from
individual models into a multi-model missing values mask; the individual model
masks are built according to common criteria: the user chooses a time window in
which missing data points are counted, and if the number of missing data points
relative to the number of total data points in a window is less than a chosen
fractional threshold, the window is discarded i.e. all the points in the window
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

Common mask for multiple models
-------------------------------

To create a combined multi-model mask (all the masks from all the analyzed
datasets combined into a single mask using a logical OR), the preprocessor
``mask_multimodel`` can be used. In contrast to ``mask_fillvalues``,
``mask_multimodel`` does not expect that the datasets have a ``time``
coordinate, but works on datasets with arbitrary (but identical) coordinates.
After ``mask_multimodel``, all involved datasets have an identical mask.

See also :func:`esmvalcore.preprocessor.mask_multimodel`.

.. _threshold_masking:

Minimum, maximum and interval masking
-------------------------------------

Thresholding on minimum and maximum accepted data values can also be performed:
masks are constructed based on the results of thresholding; inside and outside
interval thresholding and masking can also be performed. These functions are
``mask_above_threshold``, ``mask_below_threshold``, ``mask_inside_range``, and
``mask_outside_range``.

These functions always take a cube as first argument and either ``threshold``
for threshold masking or the pair ``minimum``, ``maximum`` for interval masking.

See also :func:`esmvalcore.preprocessor.mask_above_threshold` and related
functions.


.. _Horizontal regridding:

Horizontal regridding
=====================

Regridding is necessary when various datasets are available on a variety of
`lat-lon` grids and they need to be brought together on a common grid (for
various statistical operations e.g. multi-model statistics or for e.g. direct
inter-comparison or comparison with observational datasets). Regridding is
conceptually a very similar process to interpolation (in fact, the regridder
engine uses interpolation and extrapolation, with various schemes). The primary
difference is that interpolation is based on sample data points, while
regridding is based on the horizontal grid of another cube (the reference
grid). If the horizontal grids of a cube and its reference grid are sufficiently
the same, regridding is automatically and silently skipped for performance reasons.

The use of the horizontal regridding functionality is flexible depending on
what type of reference grid and what interpolation scheme is preferred. Below
we show a few examples.

Regridding on a reference dataset grid
--------------------------------------

The example below shows how to regrid on the reference dataset
``ERA-Interim`` (observational data, but just as well CMIP, obs4MIPs,
or ana4mips datasets can be used); in this case the `scheme` is
`linear`.

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
specifications are not necessarily known a priori. Regridding on an ``MxN``
cell specification is oftentimes used when operating on localized data.

.. code-block:: yaml

    preprocessors:
      regrid_preprocessor:
        regrid:
          target_grid: 2.5x2.5
          scheme: nearest

In this case the nearest-neighbor interpolation scheme is used (see below
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

Regridding to a regional target grid specification
--------------------------------------------------

This example shows how to regrid to a regional target grid specification.
This is useful if both a ``regrid`` and ``extract_region`` step are necessary.

.. code-block:: yaml

    preprocessors:
      regrid_preprocessor:
        regrid:
          target_grid:
            start_longitude: 40
            end_longitude: 60
            step_longitude: 2
            start_latitude: -10
            end_latitude: 30
            step_latitude: 2
          scheme: nearest

This defines a grid ranging from 40° to 60° longitude with 2° steps,
and -10° to 30° latitude with 2° steps. If ``end_longitude`` or ``end_latitude`` do
not fall on the grid (e.g., ``end_longitude: 61``), it cuts off at the nearest
previous value (e.g. ``60``).

The longitude coordinates will wrap around the globe if necessary, i.e.
``start_longitude: 350``, ``end_longitude: 370`` is valid input.

The arguments are defined below:

* ``start_latitude``: Latitude value of the first grid cell center (start point).
  The grid includes this value.
* ``end_latitude``: Latitude value of the last grid cell center (end point).
  The grid includes this value only if it falls on a grid point.
  Otherwise, it cuts off at the previous value.
* ``step_latitude``: Latitude distance between the centers of two neighbouring cells.
* ``start_longitude``: Latitude value of the first grid cell center (start point).
  The grid includes this value.
* ``end_longitude``: Longitude value of the last grid cell center (end point).
  The grid includes this value only if it falls on a grid point.
  Otherwise, it cuts off at the previous value.
* ``step_longitude``: Longitude distance between the centers of two neighbouring cells.

Regridding (interpolation, extrapolation) schemes
-------------------------------------------------

ESMValCore has a number of built-in regridding schemes, which are presented in
:ref:`built-in regridding schemes`. Additionally, it is also possible to use
third party regridding schemes designed for use with :doc:`Iris
<iris:index>`. This is explained in :ref:`generic regridding schemes`.

Grid types
~~~~~~~~~~

In ESMValCore, we distinguish between three grid types (note that these might
differ from other definitions):

* **Regular grid**: A rectilinear grid with 1D latitude and 1D longitude
  coordinates which are orthogonal to each other.
* **Irregular grid**: A general curvilinear grid with 2D latitude and 2D
  longitude coordinates with common dimensions.
* **Unstructured grid**: A grid with 1D latitude and 1D longitude coordinates
  with common dimensions (i.e., a simple list of points).

.. _built-in regridding schemes:

Built-in regridding schemes
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``linear``: Bilinear regridding.
  For source data on a regular grid, uses :obj:`~iris.analysis.Linear` with
  `extrapolation_mode='mask'`.
  For source data on an irregular grid, uses
  :class:`~esmvalcore.preprocessor.regrid_schemes.ESMPyLinear`.
  For source data on an unstructured grid, uses
  :class:`~esmvalcore.preprocessor.regrid_schemes.UnstructuredLinear`.
* ``nearest``: Nearest-neighbor regridding.
  For source data on a regular grid, uses :obj:`~iris.analysis.Nearest` with
  `extrapolation_mode='mask'`.
  For source data on an irregular grid, uses
  :class:`~esmvalcore.preprocessor.regrid_schemes.ESMPyNearest`.
  For source data on an unstructured grid, uses
  :class:`~esmvalcore.preprocessor.regrid_schemes.UnstructuredNearest`.
* ``area_weighted``: First-order conservative (area-weighted) regridding.
  For source data on a regular grid, uses :obj:`~iris.analysis.AreaWeighted`.
  For source data on an irregular grid, uses
  :class:`~esmvalcore.preprocessor.regrid_schemes.ESMPyAreaWeighted`.
  Source data on an unstructured grid is not supported.

.. _generic regridding schemes:

Generic regridding schemes
~~~~~~~~~~~~~~~~~~~~~~~~~~

:ref:`Iris' regridding <iris:interpolation_and_regridding>` is based around the
flexible use of so-called regridding schemes. These are classes that know how
to transform a source cube with a given grid into the grid defined by a given
target cube. Iris itself provides a number of useful schemes, but they are
largely limited to work with simple, regular grids. Other schemes can be
provided independently. This is interesting when special regridding-needs arise
or when more involved grids need to be considered. Furthermore, it may be
desirable to have finer control over the parameters of the scheme than is
afforded by the built-in schemes described above.

To facilitate this, the :func:`~esmvalcore.preprocessor.regrid` preprocessor
allows the use of any scheme designed for Iris. The scheme must be installed
and importable. To use this feature, the ``scheme`` key passed to the
preprocessor must be a dictionary instead of a simple string that contains all
necessary information. That includes a ``reference`` to the desired scheme
itself, as well as any arguments that should be passed through to the
scheme. For example, the following shows the use of the built-in scheme
:class:`iris.analysis.AreaWeighted` with a custom threshold for missing data
tolerance.

.. code-block:: yaml

    preprocessors:
      regrid_preprocessor:
        regrid:
          target_grid: 2.5x2.5
          scheme:
            reference: iris.analysis:AreaWeighted
            mdtol: 0.7

Another example is bilinear regridding with extrapolation.
This can be achieved with the :class:`iris.analysis.Linear` scheme and the
``extrapolation_mode`` keyword.
Extrapolation points will be calculated by extending the gradient of the
closest two points.

.. code-block:: yaml

    preprocessors:
      regrid_preprocessor:
        regrid:
          target_grid: 2.5x2.5
          scheme:
            reference: iris.analysis:Linear
            extrapolation_mode: extrapolate

.. note::

   Controlling the extrapolation mode allows us to avoid situations where
   extrapolating values makes little physical sense (e.g. extrapolating beyond
   the last data point).

The value of the ``reference`` key has two parts that are separated by a
``:`` with no surrounding spaces. The first part is an importable Python
module, the second refers to the scheme, i.e. some callable that will be called
with the remaining entries of the ``scheme`` dictionary passed as keyword
arguments.

One package that aims to capitalize on the :ref:`support for unstructured grids
introduced in Iris 3.2 <iris:ugrid>` is :doc:`iris-esmf-regrid:index`.
It aims to provide lazy regridding for structured regular and irregular grids,
as well as unstructured grids.
An example of its usage in a preprocessor is:

.. code-block:: yaml

    preprocessors:
      regrid_preprocessor:
        regrid:
          target_grid: 2.5x2.5
          scheme:
            reference: esmf_regrid.schemes:ESMFAreaWeighted
            mdtol: 0.7

Additionally, the use of generic schemes that take source and target grid cubes as
arguments is also supported. The call function for such schemes must be defined as
`(src_cube, grid_cube, **kwargs)` and they must return `iris.cube.Cube` objects.
The `regrid` module will automatically pass the source and grid cubes as inputs
of the scheme. An example of this usage is
the :func:`~esmf_regrid.schemes.regrid_rectilinear_to_rectilinear`
scheme available in :doc:`iris-esmf-regrid:index`:

.. code-block:: yaml

  preprocessors:
    regrid_preprocessor:
      regrid:
        target_grid: 2.5x2.5
        scheme:
          reference: esmf_regrid.schemes:regrid_rectilinear_to_rectilinear
          mdtol: 0.7

.. _caching_regridding_weights:

Reusing regridding weights
--------------------------

If desired, regridding weights can be cached to reduce run times (see `here
<https://scitools-iris.readthedocs.io/en/latest/userguide/interpolation_and_regridding.html#caching-a-regridder>`__
for technical details on this).
This can speed up the regridding of different datasets with similar source and
target grids massively, but may take up a lot of memory for extremely
high-resolution data.
By default, this feature is disabled; to enable it, use the option
``cache_weights: true`` in the preprocessor definition:

.. code-block:: yaml

    preprocessors:
      regrid_preprocessor:
        regrid:
          target_grid: 0.1x0.1
          scheme: linear
          cache_weights: true

Not all regridding schemes support weights caching. An overview of those that
do is given `here
<https://scitools-iris.readthedocs.io/en/latest/further_topics/which_regridder_to_use.html#which-regridder-to-use>`__
and in the docstrings :ref:`here <regridding_schemes>`.

See also :func:`esmvalcore.preprocessor.regrid`


.. _ensemble statistics:

Ensemble statistics
===================
For certain use cases it may be desirable to compute ensemble statistics. For
example to prevent models with many ensemble members getting excessive weight in
the multi-model statistics functions.

Theoretically, ensemble statistics are a special case (grouped) multi-model
statistics. This grouping is performed taking into account the dataset tags
`project`, `dataset`, `experiment`, and (if present) `sub_experiment`.
However, they should typically be computed earlier in the workflow.
Moreover, because multiple ensemble members of the same model are typically more
consistent/homogeneous than datasets from different models, the implementation
is more straightforward and can benefit from lazy evaluation and more efficient
computation.

The preprocessor takes a list of statistics as input:

.. code-block:: yaml

    preprocessors:
      example_preprocessor:
        ensemble_statistics:
          statistics: [mean, median]

Additional keyword arguments can be given by using a dictionary:

.. code-block:: yaml

    preprocessors:
      example_preprocessor:
        ensemble_statistics:
          statistics:
            - operator: percentile
              percent: 20
            - operator: median

This preprocessor function exposes the iris analysis package, and works with all
(capitalized) statistics from the :mod:`iris.analysis` package
that can be executed without additional arguments.
See :ref:`stat_preprocs` for more details on supported statistics.

Note that ``ensemble_statistics`` will not return the single model and ensemble files,
only the requested ensemble statistics results.

In case of wanting to save both individual ensemble members as well as the statistic results,
the preprocessor chains could be defined as:

.. code-block:: yaml

    preprocessors:
      everything_else: &everything_else
        area_statistics: ...
        regrid_time: ...
      multimodel:
        <<: *everything_else
        ensemble_statistics:

    variables:
      tas_datasets:
        short_name: tas
        preprocessor: everything_else
        ...
      tas_multimodel:
        short_name: tas
        preprocessor: multimodel
        ...


See also :func:`esmvalcore.preprocessor.ensemble_statistics`.

.. _multi-model statistics:

Multi-model statistics
======================
Computing multi-model statistics is an integral part of model analysis and
evaluation: individual models display a variety of biases depending on model
set-up, initial conditions, forcings and implementation; comparing model data to
observational data, these biases have a significantly lower statistical impact
when using a multi-model ensemble. ESMValCore has the capability of computing a
number of multi-model statistical measures: using the preprocessor module
``multi_model_statistics`` will enable the user for example to ask for either a multi-model
``mean``, ``median``, ``max``, ``min``, ``std_dev``, and / or ``percentile``
with a set of argument parameters passed to ``multi_model_statistics``.
See :ref:`stat_preprocs` for more details on supported statistics.
Percentiles can be specified with additional keyword arguments using the syntax
``statistics: [{operator: percentile, percent: xx}]``.

Restrictive computation is also available by excluding any set of models that
the user will not want to include in the statistics (by setting ``exclude:
[excluded models list]`` argument).

Input datasets may have different time coordinates.
Apart from that, all dimensions must match.
Statistics can be computed
across overlapping times only (``span: overlap``) or across the full time span
of the combined models (``span: full``). The preprocessor sets a common time
coordinate on all datasets. As the number of days in a year may vary between
calendars, (sub-)daily data with different calendars are not supported.
The preprocessor saves both the input single model files as well as the multi-model
results. In case you do not want to keep the single model files, set the
parameter ``keep_input_datasets`` to ``false`` (default value is ``true``).
To remove scalar coordinates before merging input datasets into the
multi-dataset cube, use the option ``ignore_scalar_coords: true``.
The resulting multi-dataset cube will not have scalar coordinates in this case.
This ensures that differences in scalar coordinates in the input datasets are
ignored, which is helpful if you encounter a ``ValueError: Multi-model
statistics failed to merge input cubes into a single array`` with ``Coordinates
in cube.aux_coords (scalar) differ``.
Some special scalar coordinates which are expected to differ across cubes (`p0`
and `ptop`) are always removed.

.. code-block:: yaml

    preprocessors:
      multi_model_save_input:
        multi_model_statistics:
          span: overlap
          statistics: [mean, median]
          exclude: [NCEP-NCAR-R1]
      multi_model_without_saving_input:
        multi_model_statistics:
          span: overlap
          statistics: [mean, median]
          exclude: [NCEP-NCAR-R1]
          keep_input_datasets: false
          ignore_scalar_coords: true
      multi_model_percentiles_5_95:
        multi_model_statistics:
          span: overlap
          statistics:
            - operator: percentile
              percent: 5
            - operator: percentile
              percent: 95

Multi-model statistics also supports a ``groupby`` argument. You can group by
any dataset key (``project``, ``experiment``, etc.) or a combination of keys in a list. You can
also add an arbitrary tag to a dataset definition and then group by that tag. When
using this preprocessor in conjunction with `ensemble statistics`_ preprocessor, you
can group by ``ensemble_statistics`` as well. For example:

.. code-block:: yaml

    datasets:
      - {dataset: CanESM2, exp: historical, ensemble: "r(1:2)i1p1"}
      - {dataset: CCSM4, exp: historical, ensemble: "r(1:2)i1p1"}

    preprocessors:
      example_preprocessor:
        ensemble_statistics:
          statistics: [median, mean]
        multi_model_statistics:
          span: overlap
          statistics: [min, max]
          groupby: [ensemble_statistics]
          exclude: [NCEP-NCAR-R1]

This will first compute ensemble mean and median, and then compute the multi-model
min and max separately for the ensemble means and medians. Note that this combination
will not save the individual ensemble members, only the ensemble and multimodel statistics results.

When grouping by a tag not defined in all datasets, the datasets missing the tag will
be grouped together. In the example below, datasets `UKESM` and `ERA5` would belong to the same
group, while the other datasets would belong to either ``group1`` or ``group2``

.. code-block:: yaml

    datasets:
      - {dataset: CanESM2, exp: historical, ensemble: "r(1:2)i1p1", tag: 'group1'}
      - {dataset: CanESM5, exp: historical, ensemble: "r(1:2)i1p1", tag: 'group2'}
      - {dataset: CCSM4, exp: historical, ensemble: "r(1:2)i1p1", tag: 'group2'}
      - {dataset: UKESM, exp: historical, ensemble: "r(1:2)i1p1"}
      - {dataset: ERA5}

    preprocessors:
      example_preprocessor:
        multi_model_statistics:
          span: overlap
          statistics: [min, max]
          groupby: [tag]

Note that those datasets can be excluded if listed in the ``exclude`` option.

See also :func:`esmvalcore.preprocessor.multi_model_statistics`.


.. _time operations:

Time manipulation
=================
The ``_time.py`` module contains the following preprocessor functions:

* extract_time_: Extract a time range from a cube.
* extract_season_: Extract only the times that occur within a specific season.
* extract_month_: Extract only the times that occur within a specific month.
* hourly_statistics_: Compute intra-day statistics
* daily_statistics_: Compute statistics for each day
* monthly_statistics_: Compute statistics for each month
* seasonal_statistics_: Compute statistics for each season
* annual_statistics_: Compute statistics for each year
* decadal_statistics_: Compute statistics for each decade
* climate_statistics_: Compute statistics for the full period
* resample_time_: Resample data
* resample_hours_: Convert between N-hourly frequencies by resampling
* anomalies_: Compute (standardized) anomalies
* regrid_time_: Aligns the time coordinate of each dataset, against a standardized time axis.
* timeseries_filter_: Allows application of a filter to the time-series data.
* local_solar_time_: Convert cube with UTC time to local solar time.

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
extract, i.e. DJF, MAM, JJA, SON, but also all other sequentially correct
combinations, e.g. JJAS.

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

.. _hourly_statistics:

``hourly_statistics``
---------------------

This function produces statistics at a x-hourly frequency.

Parameters:
    * `hour`: Number of hours per period.
      Must be a divisor of 24, i.e., (1, 2, 3, 4, 6, 8, 12).
    * `operator`: Operation to apply.
      See :ref:`stat_preprocs` for more details on supported statistics.
      Default is `mean`.
    * Other parameters are directly passed to the `operator` as keyword
      arguments.
      See :ref:`stat_preprocs` for more details.

See also :func:`esmvalcore.preprocessor.hourly_statistics`.

.. _daily_statistics:

``daily_statistics``
--------------------

This function produces statistics for each day in the dataset.

Parameters:
    * `operator`: Operation to apply.
      See :ref:`stat_preprocs` for more details on supported statistics.
      Default is `mean`.
    * Other parameters are directly passed to the `operator` as keyword
      arguments.
      See :ref:`stat_preprocs` for more details.

See also :func:`esmvalcore.preprocessor.daily_statistics`.

.. _monthly_statistics:

``monthly_statistics``
----------------------

This function produces statistics for each month in the dataset.

Parameters:
    * `operator`: Operation to apply.
      See :ref:`stat_preprocs` for more details on supported statistics.
      Default is `mean`.
    * Other parameters are directly passed to the `operator` as keyword
      arguments.
      See :ref:`stat_preprocs` for more details.

See also :func:`esmvalcore.preprocessor.monthly_statistics`.

.. _seasonal_statistics:

``seasonal_statistics``
-----------------------

This function produces statistics for each season (default: ``[DJF, MAM, JJA,
SON]`` or custom seasons e.g. ``[JJAS, ONDJFMAM]``) in the dataset. Note that
this function will not check for missing time points. For instance, if you are
looking at the DJF field, but your datasets starts on January 1st, the first
DJF field will only contain data from January and February.

We recommend using the extract_time to start the dataset from the following
December and remove such biased initial data points.

Parameters:
    * `operator`: Operation to apply.
      See :ref:`stat_preprocs` for more details on supported statistics.
      Default is `mean`.
    * `seasons`: Seasons to build statistics.
      Default is ``'[DJF, MAM, JJA, SON]'``.
    * Other parameters are directly passed to the `operator` as keyword
      arguments.
      See :ref:`stat_preprocs` for more details.

See also :func:`esmvalcore.preprocessor.seasonal_statistics`.

.. _annual_statistics:

``annual_statistics``
---------------------

This function produces statistics for each year.

Parameters:
    * `operator`: Operation to apply.
      See :ref:`stat_preprocs` for more details on supported statistics.
      Default is `mean`.
    * Other parameters are directly passed to the `operator` as keyword
      arguments.
      See :ref:`stat_preprocs` for more details.

See also :func:`esmvalcore.preprocessor.annual_statistics`.

.. _decadal_statistics:

``decadal_statistics``
----------------------

This function produces statistics for each decade.

Parameters:
    * `operator`: Operation to apply.
      See :ref:`stat_preprocs` for more details on supported statistics.
      Default is `mean`.
    * Other parameters are directly passed to the `operator` as keyword
      arguments.
      See :ref:`stat_preprocs` for more details.

See also :func:`esmvalcore.preprocessor.decadal_statistics`.

.. _climate_statistics:

``climate_statistics``
----------------------

This function produces statistics for the whole dataset. It can produce scalars
(if the full period is chosen) or hourly, daily, monthly or seasonal
statistics.

Parameters:
    * `operator`: Operation to apply.
      See :ref:`stat_preprocs` for more details on supported statistics.
      Default is `mean`.
    * `period`: Define the granularity of the statistics: get values for the
      full period, for each month, day of year or hour of day.
      Available periods: `full`, `season`, `seasonal`, `monthly`, `month`,
      `mon`, `daily`, `day`, `hourly`, `hour`, `hr`. Default is `full`.
    * `seasons`: if period 'seasonal' or 'season' allows to set custom seasons.
      Default is ``'[DJF, MAM, JJA, SON]'``.
    * Other parameters are directly passed to the `operator` as keyword
      arguments.
      See :ref:`stat_preprocs` for more details.

.. note::
   Some operations are weighted by the time coordinate by default, i.e., the
   length of the time intervals.
   See :ref:`stat_preprocs` for more details on supported statistics.
   For `sum`, the units of the resulting cube are multiplied by the
   corresponding time units (e.g., days).

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

    * 80% percentile for each month:

        .. code-block:: yaml

            climate_statistics:
              period: month
              operator: percentile
              percent: 80

See also :func:`esmvalcore.preprocessor.climate_statistics`.

.. _resample_time:

``resample_time``
-----------------

This function changes the frequency of the data in the cube by extracting the
timesteps that meet the criteria. It is important to note that it is mainly
meant to be used with instantaneous data.

Parameters:
    * month: Extract only timesteps from the given month or do nothing if None.
      Default is `None`
    * day: Extract only timesteps from the given day of month or do nothing if
      None. Default is `None`
    * hour: Extract only timesteps from the given hour or do nothing if None.
      Default is `None`

Examples:
    * Hourly data to daily:

        .. code-block:: yaml

            resample_time:
              hour: 12

    * Hourly data to monthly:

        .. code-block:: yaml

            resample_time:
              hour: 12
              day: 15

    * Daily data to monthly:

        .. code-block:: yaml

            resample_time:
              day: 15

See also :func:`esmvalcore.preprocessor.resample_time`.


resample_hours:

``resample_hours``
------------------

This function changes the frequency of the data in the cube by extracting the
timesteps that belongs to the desired frequency. It is important to note that
it is mainly mean to be used with instantaneous data

Parameters:
    * interval: New frequency of the data. Must be a divisor of 24
    * offset: First desired hour. Default 0. Must be lower than the interval

Examples:
    * Convert to 12-hourly, by getting timesteps at 0:00 and 12:00:

        .. code-block:: yaml

            resample_hours:
              hours: 12

    * Convert to 12-hourly, by getting timesteps at 6:00 and 18:00:

        .. code-block:: yaml

            resample_hours:
              hours: 12
	      offset: 6

See also :func:`esmvalcore.preprocessor.resample_hours`.

.. _anomalies:

``anomalies``
----------------------

This function computes the anomalies for the whole dataset. It can compute
anomalies from the full, seasonal, monthly, daily and hourly climatologies.
Optionally standardized anomalies can be calculated.

Parameters:
    * period: define the granularity of the climatology to use:
      full period, seasonal, monthly, daily or hourly.
      Available periods: 'full', 'season', 'seasonal', 'monthly', 'month',
      'mon', 'daily', 'day', 'hourly', 'hour', 'hr'. Default is 'full'
    * reference: Time slice to use as the reference to compute the climatology
      on. Can be 'null' to use the full cube or a dictionary with the
      parameters from extract_time_. Default is null
    * standardize: if true calculate standardized anomalies (default: false)
    * seasons: if period 'seasonal' or 'season' allows to set custom seasons.
      Default is '[DJF, MAM, JJA, SON]'
Examples:
    * Anomalies from the full period climatology:

        .. code-block:: yaml

            anomalies:

    * Anomalies from the full period monthly climatology:

        .. code-block:: yaml

            anomalies:
              period: month

    * Standardized anomalies from the full period climatology:

        .. code-block:: yaml

            anomalies:
              standardized: true


     * Standardized Anomalies from the 1979-2000 monthly climatology:

        .. code-block:: yaml

            anomalies:
              period: month
              reference:
                start_year: 1979
                start_month: 1
                start_day: 1
                end_year: 2000
                end_month: 12
                end_day: 31
              standardize: true

See also :func:`esmvalcore.preprocessor.anomalies`.


.. _regrid_time:

``regrid_time``
---------------

This function aligns the time points and bounds of an input dataset according
to the following rules:

* Decadal data: 1 January 00:00:00 for the given year.
  Example: 1 January 2005 00:00:00 for given year 2005 (decade 2000-2010).
* Yearly data: 1 July 00:00:00 for each year.
  Example: 1 July 1993 00:00:00 for the year 1993.
* Monthly data: 15th day 00:00:00 for each month.
  Example: 15 October 1993 00:00:00 for the month October 1993.
* Daily data: 12:00:00 for each day.
  Example: 14 March 1996 12:00:00 for the day 14 March 1996.
* `n`-hourly data where `n` is a divisor of 24: center of each time interval.
  Example: 03:00:00 for interval 00:00:00-06:00:00 (6-hourly data), 16:30:00
  for interval 15:00:00-18:00:00 (3-hourly data), or 09:30:00 for interval
  09:00:00-10:00:00 (hourly data).

The frequency of the input data is automatically determined from the CMOR table
of the corresponding variable, but can be overwritten in the recipe if
necessary.
This function does not alter the data in any way.

.. note::

  By default, this preprocessor will not change the calendar of the input time
  coordinate.
  For decadal, yearly, and monthly data, it is possible to change the calendar
  using the optional `calendar` argument.
  Be aware that changing the calendar might introduce (small) errors to your
  data, especially for extensive quantities (those that depend on the period
  length).

Parameters:
    * `frequency`: Data frequency.
      If not given, use the one from the CMOR tables of the corresponding
      variable.
    * `calendar`: If given, transform the calendar to the one specified
      (examples: `standard`, `365_day`, etc.).
      This only works for decadal, yearly and monthly data, and will raise an
      error for other frequencies.
      If not set, the calendar will not be changed.
    * `units` (default: `days since 1850-01-01 00:00:00`): Reference time units
      used if the calendar of the data is changed.
      Ignored if `calendar` is not set.

Examples:

Change the input calendar to `standard` and use custom units:

.. code-block:: yaml

  regrid_time:
    calendar: standard
    units: days since 2000-01-01

See also :func:`esmvalcore.preprocessor.regrid_time`.


.. _timeseries_filter:

``timeseries_filter``
---------------------

This function allows the user to apply a filter to the timeseries data. This filter may be
of the user's choice (currently only the ``low-pass`` Lanczos filter is implemented); the
implementation is inspired by this `iris example
<https://scitools-iris.readthedocs.io/en/latest/generated/gallery/general/plot_SOI_filtering.html>`_ and uses aggregation via :obj:`iris.cube.Cube.rolling_window`.

Parameters:
    * window: the length of the filter window (in units of cube time coordinate).
    * span: period (number of months/days, depending on data frequency) on which
      weights should be computed e.g. for 2-yearly: span = 24 (2 x 12 months).
      Make sure span has the same units as the data cube time coordinate.
    * filter_type: the type of filter to be applied; default 'lowpass'.
      Available types: 'lowpass'.
    * filter_stats: the type of statistic to aggregate on the rolling window;
      default 'sum'. Available operators: 'mean', 'median', 'std_dev', 'sum', 'min', 'max', 'rms'.

Examples:
    * Lowpass filter with a monthly mean as operator:

        .. code-block:: yaml

            timeseries_filter:
                window: 3  # 3-monthly filter window
                span: 12   # weights computed on the first year
                filter_type: lowpass  # low-pass filter
                filter_stats: mean    # 3-monthly mean lowpass filter

See also :func:`esmvalcore.preprocessor.timeseries_filter`.

.. _local_solar_time:

``local_solar_time``
--------------------

Many variables in the Earth system show a strong diurnal cycle.
The reason for that is of course Earth's rotation around its own axis, which
leads to a diurnal cycle of the incoming solar radiation.
While UTC time is a very good absolute time measure, it is not really suited to
analyze diurnal cycles over larger regions.
For example, diurnal cycles over Russia and the USA are phase-shifted by ~180°
= 12 hr in UTC time.

This is where the `local solar time (LST)
<https://en.wikipedia.org/wiki/Solar_time>`__ comes into play:
For a given location, 12:00 noon LST is defined as the moment when the sun
reaches its highest point in the sky.
By using this definition based on the origin of the diurnal cycle (the sun), we
can directly compare diurnal cycles across the globe.
LST is mainly determined by the longitude of a location, but due to the
eccentricity of Earth's orbit, it also depends on the day of year (see
`equation of time <https://en.wikipedia.org/wiki/Equation_of_time>`__).
However, this correction is at most ~15 min, which is usually smaller than the
highest frequency output of CMIP6 models (1 hr) and smaller than the time scale
for diurnal evolution of meteorological phenomena (which is in the order of
hours, not minutes).
Thus, instead, we use the **mean** LST, which solely depends on longitude:

.. math::

  LST = UTC + 12 \cdot \frac{lon}{180°}

where the times are given in hours and `lon` in degrees in the interval [-180,
180].
To transform data from UTC to LST, this preprocessor shifts data along the time
axis based on the longitude.

This preprocessor does not need any additional parameters.

Example:

.. code-block:: yaml

  calculate_local_solar_time:
    local_solar_time:

See also :func:`esmvalcore.preprocessor.local_solar_time`.


.. _area operations:

Area manipulation
=================
The area manipulation module contains the following preprocessor functions:

* extract_coordinate_points_: Extract a point with arbitrary coordinates given an interpolation scheme.
* extract_region_: Extract a region from a cube based on ``lat/lon``
  corners.
* extract_named_regions_: Extract a specific region from in the region
  coordinate.
* extract_shape_: Extract a region defined by a shapefile.
* extract_point_: Extract a single point (with interpolation)
* extract_location_: Extract a single point by its location (with interpolation)
* zonal_statistics_: Compute zonal statistics.
* meridional_statistics_: Compute meridional statistics.
* area_statistics_: Compute area statistics.


``extract_coordinate_points``
-----------------------------

This function extracts points with given coordinates, following either a
``linear`` or a ``nearest`` interpolation scheme.
The resulting point cube will match the respective coordinates to
those of the input coordinates. If the input coordinate is a scalar,
the dimension will be a scalar in the output cube.

If the point to be extracted has at least one of the coordinate point
values outside the interval of the cube's same coordinate values, then
no extrapolation will be performed, and the resulting extracted cube
will have fully masked data.

Examples:
    * Extract a point from coordinate `grid_latitude` with given coordinate value 26.0:

        .. code-block:: yaml

            extract_coordinate_points:
              definition:
                grid_latitude: 26.
              scheme: nearest

See also :func:`esmvalcore.preprocessor.extract_coordinate_points`.

.. _extract_region:

``extract_region``
------------------

This function returns a subset of the data on the rectangular region requested.
The boundaries of the region are provided as latitude and longitude coordinates
in the arguments:

* ``start_longitude``
* ``end_longitude``
* ``start_latitude``
* ``end_latitude``

Note that this function can only be used to extract a rectangular region. Use
``extract_shape`` to extract any other shaped region from a shapefile.

If the grid is irregular, the returned region retains the original coordinates,
but is cropped to a rectangular bounding box defined by the start/end
coordinates. The deselected area inside the region is masked.

See also :func:`esmvalcore.preprocessor.extract_region`.


``extract_named_regions``
-------------------------

This function extracts a specific named region from the data. This function
takes the following argument: ``regions`` which is either a string or a list
of strings of named regions. Note that the dataset must have a ``region``
coordinate which includes a list of strings as values. This function then
matches the named regions against the requested string.

See also :func:`esmvalcore.preprocessor.extract_named_regions`.


``extract_shape``
-----------------

Extract a shape or a representative point for this shape from the data.

Parameters:
  * ``shapefile``: path to the shapefile containing the geometry of the
    region to be extracted.
    If the file contains multiple shapes behaviour depends on the
    ``decomposed`` parameter.
    This path can be relative to ``auxiliary_data_dir`` defined in the
    :ref:`user configuration file` or relative to
    ``esmvalcore/preprocessor/shapefiles`` (in that priority order).
    Alternatively, a string (see "Shapefile name" below) can be given to load
    one of the following shapefiles that are shipped with ESMValCore:

    =============== ===================== ==========================================
    Shapefile name  Description           Reference
    =============== ===================== ==========================================
    ar6             IPCC WG1 reference    https://doi.org/10.5281/zenodo.5176260
                    regions (v4) used in
                    Assessment Report 6
    =============== ===================== ==========================================

  * ``method``: the method to select the region, selecting either all points
    contained by the shape or a single representative point.
    Choose either `'contains'` or `'representative'`.
    If not a single grid point is contained in the shape, a representative
    point will be selected.
  * ``crop``: by default extract_region_ will be used to crop the data to a
    minimal rectangular region containing the shape.
    Set to ``false`` to only mask data outside the shape.
    Data on irregular grids will not be cropped.
  * ``decomposed``: by default ``false``; in this case the union of all the
    regions in the shapefile is masked out.
    If set to ``true``, the regions in the shapefiles are masked out separately
    and the output cube will have an additional dimension ``shape_id``
    describing the requested regions.
  * ``ids``: Shapes to be read from the shapefile.
    Can be given as:

    * :obj:`list`: IDs are assigned from the attributes ``name``, ``NAME``,
      ``Name``, ``id``, or ``ID`` (in that priority order; the first one
      available is used).
      If none of these attributes are available in the shapefile,
      assume that the given `ids` correspond to the reading order of the
      individual shapes.
      So, for example, if a file has both ``name`` and ``id`` attributes, the
      ids will be assigned from ``name``.
      If the file only has the ``id`` attribute, it will be taken from it and
      if no ``name`` nor ``id`` attributes are present, an integer ID starting
      from 0 will be assigned automatically when reading the shapes.
      We discourage to rely on this last behaviour as we can not assure that
      the reading order will be the same on different platforms, so we
      encourage you to specify a custom attribute using a :obj:`dict` (see
      below) instead.
      Note: An empty list is interpreted as ``ids=None`` (see below).
    * :obj:`dict`: IDs (dictionary value; :obj:`list` of :obj:`str`) are
      assigned from attribute given as dictionary key (:obj:`str`).
      Only dictionaries with length 1 are supported.
      Example: ``ids={'Acronym': ['GIC', 'WNA']}``.
    * `None`: select all available regions from the shapefile.

Examples:
    * Extract the shape of the river Elbe from a shapefile:

        .. code-block:: yaml

            extract_shape:
              shapefile: Elbe.shp
              method: contains

    * Extract the shape of several countries:

        .. code-block:: yaml

            extract_shape:
            shapefile: NaturalEarth/Countries/ne_110m_admin_0_countries.shp
            decomposed: True
            method: contains
            ids:
              - Spain
              - France
              - Italy
              - United Kingdom
              - Taiwan

    * Extract European AR6 regions:

        .. code-block:: yaml

            extract_shape:
              shapefile: ar6
              method: contains
              ids:
                Acronym:
                  - NEU
                  - WCE
                  - MED

See also :func:`esmvalcore.preprocessor.extract_shape`.


``extract_point``
-----------------

Extract a single point from the data. This is done using either
nearest or linear interpolation.

Returns a cube with the extracted point(s), and with adjusted latitude
and longitude coordinates (see below).

Multiple points can also be extracted, by supplying an array of
latitude and/or longitude coordinates. The resulting point cube will
match the respective latitude and longitude coordinate to those of the
input coordinates. If the input coordinate is a scalar, the dimension
will be missing in the output cube (that is, it will be a scalar).

If the point to be extracted has at least one of the coordinate point
values outside the interval of the cube's same coordinate values, then
no extrapolation will be performed, and the resulting extracted cube
will have fully masked data.

Parameters:
  * ``cube``: the input dataset cube.
  * ``latitude``, ``longitude``: coordinates (as floating point
    values) of the point to be extracted. Either (or both) can also
    be an array of floating point values.
  * ``scheme``: interpolation scheme: either ``'linear'`` or
    ``'nearest'``. There is no default.

See also :func:`esmvalcore.preprocessor.extract_point`.


.. _extract_location:

``extract_location``
--------------------

Extract a single point using a location name, with interpolation
(either linear or nearest). This preprocessor extracts a single
location point from a cube, according to the given interpolation
scheme ``scheme``. The function retrieves the coordinates of the
location and then calls the :func:`esmvalcore.preprocessor.extract_point`
preprocessor. It can be used to locate cities and villages,
but also mountains or other geographical locations.

.. note::
   Note that this function's geolocator application needs a
   working internet connection.

Parameters:
    * `cube`: the input dataset cube to extract a point from.
    * `location`: the reference location. Examples: 'mount everest',
      'romania', 'new york, usa'. Raises ValueError if none supplied.
    * `scheme` : interpolation scheme. `linear` or `nearest`.
      There is no default, raises ValueError if none supplied.

See also :func:`esmvalcore.preprocessor.extract_location`.


``zonal_statistics``
--------------------

The function calculates the zonal statistics by applying an operator
along the longitude coordinate.

Parameters:
    * `operator`: Operation to apply.
      See :ref:`stat_preprocs` for more details on supported statistics.
    * `normalize`: If given, do not return the statistics cube itself, but
      rather, the input cube, normalized with the statistics cube. Can either
      be `subtract` (statistics cube is subtracted from the input cube) or
      `divide` (input cube is divided by the statistics cube).
    * Other parameters are directly passed to the `operator` as keyword
      arguments.
      See :ref:`stat_preprocs` for more details.

See also :func:`esmvalcore.preprocessor.zonal_statistics`.


``meridional_statistics``
-------------------------

The function calculates the meridional statistics by applying an
operator along the latitude coordinate. This function takes one
argument:

Parameters:
    * `operator`: Operation to apply.
      See :ref:`stat_preprocs` for more details on supported statistics.
    * `normalize`: If given, do not return the statistics cube itself, but
      rather, the input cube, normalized with the statistics cube. Can either
      be `subtract` (statistics cube is subtracted from the input cube) or
      `divide` (input cube is divided by the statistics cube).
    * Other parameters are directly passed to the `operator` as keyword
      arguments.
      See :ref:`stat_preprocs` for more details.

See also :func:`esmvalcore.preprocessor.meridional_statistics`.


.. _area_statistics:

``area_statistics``
-------------------

This function can be used to apply several different operations in the
horizontal plane: for example, mean, sum, standard deviation, median, variance,
minimum, maximum and root mean square.
Some operations are grid cell area weighted by default.
For sums, the units of the resulting cubes are multiplied by m :math:`^2`.
See :ref:`stat_preprocs` for more details on supported statistics.

Note that this function is applied over the entire dataset.
If only a specific region, depth layer or time period is required, then those
regions need to be removed using other preprocessor operations in advance.

For weighted statistics, this function requires a cell area `cell measure`_,
unless the coordinates of the input data are regular 1D latitude and longitude
coordinates so the cell areas can be computed internally.
The required supplementary variable, either ``areacella`` for atmospheric
variables or ``areacello`` for ocean variables, can be attached to the main
dataset as described in :ref:`supplementary_variables`.

Parameters:
    * `operator`: Operation to apply.
      See :ref:`stat_preprocs` for more details on supported statistics.
    * `normalize`: If given, do not return the statistics cube itself, but
      rather, the input cube, normalized with the statistics cube. Can either
      be `subtract` (statistics cube is subtracted from the input cube) or
      `divide` (input cube is divided by the statistics cube).
    * Other parameters are directly passed to the `operator` as keyword
      arguments.
      See :ref:`stat_preprocs` for more details.

Examples:
* Calculate global mean:

  .. code-block:: yaml

    area_statistics:
      operator: mean

* Subtract global mean from dataset:

  .. code-block:: yaml

    area_statistics:
      operator: mean
      normalize: subtract

See also :func:`esmvalcore.preprocessor.area_statistics`.


.. _volume operations:

Volume manipulation
===================
The ``_volume.py`` module contains the following preprocessor functions:

* ``axis_statistics``: Perform operations along a given axis.
* ``extract_volume``: Extract a specific depth range from a cube.
* ``volume_statistics``: Calculate the volume-weighted average.
* ``depth_integration``: Integrate over the depth dimension.
* ``extract_transect``: Extract data along a line of constant latitude or
  longitude.
* ``extract_trajectory``: Extract data along a specified trajectory.


``extract_volume``
------------------

Extract a specific range in the `z`-direction from a cube. The range is given as an interval
that can be:

* open ``(z_min, z_max)``, in which the extracted range does not include ``z_min`` nor ``z_max``.
* closed ``[z_min, z_max]``, in which the extracted includes both ``z_min`` and ``z_max``.
* left closed ``[z_min, z_max)``, in which the extracted range includes ``z_min`` but not ``z_max``.
* right closed ``(z_min, z_max]``, in which the extracted range includes ``z_max`` but not ``z_min``.

The extraction is performed by applying a constraint on the coordinate values, without any kind of interpolation.

This function takes four arguments:

* ``z_min``  to define the minimum value of the range to extract in the `z`-direction.
* ``z_max`` to define the maximum value of the range to extract in the `z`-direction.
* ``interval_bounds`` to define whether the bounds of the interval are ``open``, ``closed``,
    ``left_closed`` or ``right_closed``. Default is ``open``.
* ``nearest_value`` to extract a range taking into account the values of the z-coordinate that
    are closest to ``z_min`` and ``z_max``. Default is ``False``.

As the coordinate points are likely to vary depending on the dataset, sometimes it might be
useful to adjust the given ``z_min`` and ``z_max`` values to the values of the coordinate
points before performing an extraction.  This behaviour can be achieved by setting the
``nearest_value`` argument to ``True``.

For example, in a cube with ``z_coord = [0., 1.5, 2.6., 3.8., 5.4]``, the preprocessor below:

.. code-block:: yaml

  preprocessors:
    extract_volume:
      z_min: 1.
      z_max: 5.
      interval_bounds: 'closed'

would return a cube with a ``z_coord`` defined as ``z_coord = [1.5, 2.6., 3.8.]``,
since these are the values that strictly fall into the range given by ``[z_min=1, z_max=5]``.

Whereas setting ``ǹearest_value: True``:

.. code-block:: yaml

  preprocessors:
    extract_volume:
      z_min: 1.
      z_max: 5.
      interval_bounds: 'closed'
      nearest_value: True

would return a cube with a ``z_coord`` defined as ``z_coord = [1.5, 2.6., 3.8., 5.4]``,
since ``z_max = 5`` is closest to the coordinate point ``z = 5.4`` than it is to ``z = 3.8``.

Note that this preprocessor requires the requested `z`-coordinate range to be the same sign
as the Iris cube. That is, if the cube has `z`-coordinate as negative, then
``z_min`` and ``z_max`` need to be negative numbers.

See also :func:`esmvalcore.preprocessor.extract_volume`.


.. _volume_statistics:

``volume_statistics``
---------------------

This function calculates the volume-weighted average across three dimensions,
but maintains the time dimension.

By default, the `mean` operation is weighted by the grid cell volumes.

For weighted statistics, this function requires a cell volume `cell measure`_,
unless it has a cell_area `cell measure`_ or the coordinates of the input data
are regular 1D latitude and longitude coordinates so the cell volumes can be
computed internally.
The required supplementary variable ``volcello``, or ``areacello`` in its
absence, can be attached to the main dataset as described in
:ref:`supplementary_variables`.

No depth coordinate is required as this is determined by Iris. However, to
compute the volume automatically when ``volcello`` is not provided, the depth
coordinate units should be convertible to meters.

Parameters:
    * `operator`: Operation to apply.
      At the moment, only `mean` is supported.
      See :ref:`stat_preprocs` for more details on supported statistics.
    * `normalize`: If given, do not return the statistics cube itself, but
      rather, the input cube, normalized with the statistics cube. Can either
      be `subtract` (statistics cube is subtracted from the input cube) or
      `divide` (input cube is divided by the statistics cube).
    * Other parameters are directly passed to the `operator` as keyword
      arguments.
      See :ref:`stat_preprocs` for more details.

See also :func:`esmvalcore.preprocessor.volume_statistics`.

.. _axis_statistics:

``axis_statistics``
---------------------

This function operates over a given axis, and removes it from the
output cube.

Takes arguments:
    * `axis`: direction over which the statistics will be performed.
      Possible values for the axis are `x`, `y`, `z`, `t`.
    * `operator`: Operation to apply.
      See :ref:`stat_preprocs` for more details on supported statistics.
    * `normalize`: If given, do not return the statistics cube itself, but
      rather, the input cube, normalized with the statistics cube. Can either
      be `subtract` (statistics cube is subtracted from the input cube) or
      `divide` (input cube is divided by the statistics cube).
    * Other parameters are directly passed to the `operator` as keyword
      arguments.
      See :ref:`stat_preprocs` for more details.

.. note::
   The coordinate associated to the axis over which the operation will
   be performed must be one-dimensional, as multidimensional coordinates
   are not supported in this preprocessor.

   Some operations are weighted by the corresponding coordinate bounds by
   default.
   For sums, the units of the resulting cubes are multiplied by the
   corresponding coordinate units.
   See :ref:`stat_preprocs` for more details on supported statistics.

See also :func:`esmvalcore.preprocessor.axis_statistics`.


``depth_integration``
---------------------

This function integrates over the depth dimension.
This function does a weighted sum along the `z`-coordinate, and removes the `z`
direction of the output cube.
This preprocessor takes no arguments.
The units of the resulting cube are multiplied by the `z`-coordinate units.

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
The three arguments are: ``latitudes``, ``longitudes`` and number of point
needed for extrapolation ``number_points``.

If two points are provided, the ``number_points`` argument is used to set a
the number of places to extract between the two end points.

If more than two points are provided, then ``extract_trajectory`` will produce
a cube which has extrapolated the data of the cube to those points, and
``number_points`` is not needed.

Note that this function uses the expensive ``interpolate`` method from
``Iris.analysis.trajectory``, but it may be necessary for irregular grids.

See also :func:`esmvalcore.preprocessor.extract_trajectory`.


.. _cycles:

Cycles
======

The ``_cycles.py`` module contains the following preprocessor functions:

* ``amplitude``: Extract the peak-to-peak amplitude of a cycle aggregated over
  specified coordinates.

``amplitude``
-------------

This function extracts the peak-to-peak amplitude (maximum value minus minimum
value) of a field aggregated over specified coordinates. Its only argument is
``coords``, which can either be a single coordinate (given as :obj:`str`) or
multiple coordinates (given as :obj:`list` of :obj:`str`). Usually, these
coordinates refer to temporal categorised coordinates
:obj:`iris.coord_categorisation`
like `year`, `month`, `day of year`, etc. For example, to extract the amplitude
of the annual cycle for every single year in the data, use ``coords: year``; to
extract the amplitude of the diurnal cycle for every single day in the data,
use ``coords: [year, day_of_year]``.

See also :func:`esmvalcore.preprocessor.amplitude`.


.. _trend:

Trend
=====

The trend module contains the following preprocessor functions:

* ``linear_trend``: Calculate linear trend along a specified coordinate.
* ``linear_trend_stderr``: Calculate standard error of linear trend along a
  specified coordinate.

``linear_trend``
----------------

This function calculates the linear trend of a dataset (defined as slope of an
ordinary linear regression) along a specified coordinate. The only argument of
this preprocessor is ``coordinate`` (given as :obj:`str`; default value is
``'time'``).

See also :func:`esmvalcore.preprocessor.linear_trend`.

``linear_trend_stderr``
-----------------------

This function calculates the standard error of the linear trend of a dataset
(defined as the standard error of the slope in an ordinary linear regression)
along a specified coordinate. The only argument of this preprocessor is
``coordinate`` (given as :obj:`str`; default value is ``'time'``). Note that
the standard error is **not** identical to a confidence interval.

See also :func:`esmvalcore.preprocessor.linear_trend_stderr`.


.. _detrend:

Detrend
=======

ESMValCore also supports detrending along any dimension using
the preprocessor function 'detrend'.
This function has two parameters:

* ``dimension``: dimension to apply detrend on. Default: "time"
* ``method``: It can be ``linear`` or ``constant``. Default: ``linear``

If method is ``linear``, detrend will calculate the linear trend along the
selected axis and subtract it to the data. For example, this can be used to
remove the linear trend caused by climate change on some variables is selected
dimension is time.

If method is ``constant``, detrend will compute the mean along that dimension
and subtract it from the data

See also :func:`esmvalcore.preprocessor.detrend`.

.. _rolling window statistics:

Rolling window statistics
=========================

One can calculate rolling window statistics using the
preprocessor function ``rolling_window_statistics``.
This function takes three parameters:

    * `coordinate`: Coordinate over which the rolling-window statistics is
      calculated.
    * `operator`: Operation to apply.
      See :ref:`stat_preprocs` for more details on supported statistics.
    * `window_length`: size of the rolling window to use (number of points).
    * Other parameters are directly passed to the `operator` as keyword
      arguments.
      See :ref:`stat_preprocs` for more details.

This example applied on daily precipitation data calculates two-day rolling
precipitation sum.

.. code-block:: yaml

  preprocessors:
    preproc_rolling_window:
      coordinate: time
      operator: sum
      window_length: 2

See also :func:`esmvalcore.preprocessor.rolling_window_statistics`.


.. _unit conversion:

Unit conversion
===============

``convert_units``
-----------------

Converting units is also supported. This is particularly useful in
cases where different datasets might have different units, for example
when comparing CMIP5 and CMIP6 variables where the units have changed
or in case of observational datasets that are delivered in different
units.

In these cases, having a unit conversion at the end of the processing
will guarantee homogeneous input for the diagnostics.

Conversion is only supported between compatible units!
In other words, converting temperature units from ``degC`` to ``Kelvin`` works
fine, while changing units from ``kg`` to ``m`` will not work.

However, there are some well-defined exceptions from this rule in order to
transform one quantity to another (physically related) quantity.
These quantities are identified via their ``standard_name`` and their ``units``
(units convertible to the ones defined are also supported).
For example, this enables conversions between precipitation fluxes measured in
``kg m-2 s-1`` and precipitation rates measured in ``mm day-1`` (and vice
versa).
Currently, the following special conversions are supported:

* ``precipitation_flux`` (``kg m-2 s-1``) --
  ``lwe_precipitation_rate`` (``mm day-1``)
* ``equivalent_thickness_at_stp_of_atmosphere_ozone_content`` (``m``) --
  ``equivalent_thickness_at_stp_of_atmosphere_ozone_content`` (``DU``)

.. hint::
   Names in the list correspond to ``standard_names`` of the input data.
   Conversions are allowed from each quantity to any other quantity given in a
   bullet point.
   The corresponding target quantity is inferred from the desired target units.
   In addition, any other units convertible to the ones given are also
   supported (e.g., instead of ``mm day-1``, ``m s-1`` is also supported).

.. note::
   For the transformation between the different precipitation variables, a
   water density of ``1000 kg m-3`` is assumed.

See also :func:`esmvalcore.preprocessor.convert_units`.


``accumulate_coordinate``
-------------------------

This function can be used to weight data using the bounds from a given coordinate.
The resulting cube will then have units given by ``cube_units * coordinate_units``.

For instance, if a variable has units such as ``X s-1``, using ``accumulate_coordinate``
on the time coordinate would result on a cube where the data would be multiplied
by the time bounds and the resulting units for the variable would be converted to ``X``.
In this case, weighting the data with the time coordinate would allow to cancel
the time units in the variable.

.. note::
   The coordinate used to weight the data must be one-dimensional, as multidimensional
   coordinates are not supported in this preprocessor.


See also :func:`esmvalcore.preprocessor.accumulate_coordinate.`


.. _comparison_with_ref:

Comparison with reference dataset
=================================

This module contains the following preprocessor functions:

* ``bias``: Calculate absolute or relative biases with respect to a reference
  dataset.
* ``distance_metric``: Calculate absolute or relative biases with respect to a
  reference dataset.

``bias``
--------

This function calculates biases with respect to a given reference dataset.
For this, exactly one input dataset needs to be declared as
``reference_for_bias: true`` in the recipe, e.g.,

.. code-block:: yaml

  datasets:
    - {dataset: CanESM5, project: CMIP6, ensemble: r1i1p1f1, grid: gn}
    - {dataset: CESM2,   project: CMIP6, ensemble: r1i1p1f1, grid: gn}
    - {dataset: MIROC6,  project: CMIP6, ensemble: r1i1p1f1, grid: gn}
    - {dataset: ERA-Interim, project: OBS6, tier: 3, type: reanaly, version: 1,
       reference_for_bias: true}

In the example above, ERA-Interim is used as reference dataset for the bias
calculation.
The reference dataset needs to be broadcastable to all other datasets.
This supports `iris' rich broadcasting abilities
<https://scitools-iris.readthedocs.io/en/stable/userguide/cube_maths.
html#calculating-a-cube-anomaly>`__.
To ensure this, the preprocessors :func:`esmvalcore.preprocessor.regrid` and/or
:func:`esmvalcore.preprocessor.regrid_time` might be helpful.

The ``bias`` preprocessor supports 4 optional arguments in the recipe:

* ``bias_type`` (:obj:`str`, default: ``'absolute'``): Bias type that is
  calculated. Can be ``'absolute'`` (i.e., calculate bias for dataset
  :math:`X` and reference :math:`R` as :math:`X - R`) or ``relative`` (i.e.,
  calculate bias as :math:`\frac{X - R}{R}`).
* ``denominator_mask_threshold`` (:obj:`float`, default: ``1e-3``):
  Threshold to mask values close to zero in the denominator (i.e., the
  reference dataset) during the calculation of relative biases. All values
  in the reference dataset with absolute value less than the given threshold
  are masked out. This setting is ignored when ``bias_type`` is set to
  ``'absolute'``. Please note that for some variables with very small
  absolute values (e.g., carbon cycle fluxes, which are usually :math:`<
  10^{-6}` kg m :math:`^{-2}` s :math:`^{-1}`) it is absolutely essential to
  change the default value in order to get reasonable results.
* ``keep_reference_dataset`` (:obj:`bool`, default: ``False``): If
  ``True``, keep the reference dataset in the output. If ``False``, drop the
  reference dataset.
* ``exclude`` (:obj:`list` of :obj:`str`): Exclude specific datasets from
  this preprocessor. Note that this option is only available in the recipe,
  not when using :func:`esmvalcore.preprocessor.bias` directly (e.g., in
  another python script). If the reference dataset has been excluded, an
  error is raised.

Example:

.. code-block:: yaml

    preprocessors:
      preproc_bias:
        bias:
          bias_type: relative
          denominator_mask_threshold: 1e-8
          keep_reference_dataset: true
          exclude: [CanESM2]

See also :func:`esmvalcore.preprocessor.bias`.

.. _distance_metric:

``distance_metric``
-------------------

This function calculates a distance metric with respect to a given reference
dataset.
For this, exactly one input dataset needs to be declared as
``reference_for_metric: true`` in the recipe, e.g.,

.. code-block:: yaml

  datasets:
    - {dataset: CanESM5, project: CMIP6, ensemble: r1i1p1f1, grid: gn}
    - {dataset: CESM2,   project: CMIP6, ensemble: r1i1p1f1, grid: gn}
    - {dataset: MIROC6,  project: CMIP6, ensemble: r1i1p1f1, grid: gn}
    - {dataset: ERA-Interim, project: OBS6, tier: 3, type: reanaly, version: 1,
       reference_for_metric: true}

In the example above, ERA-Interim is used as reference dataset for the distance
metric calculation.
All datasets need to have the same shape and coordinates.
To ensure this, the preprocessors :func:`esmvalcore.preprocessor.regrid` and/or
:func:`esmvalcore.preprocessor.regrid_time` might be helpful.

The ``distance_metric`` preprocessor supports the following arguments in the
recipe:

.. _list_of_distance_metrics:

* ``metric`` (:obj:`str`): Distance metric that is calculated.
  Must be one of

  * ``'rmse'``: `Unweighted root mean square error`_.

  .. math::

    RMSE = \sqrt{\frac{1}{N} \sum_{i=1}^N \left( x_i - r_i \right)^2}

  * ``'weighted_rmse'``: `Weighted root mean square error`_.

  .. math::

    WRMSE = \sqrt{\sum_{i=1}^N w_i \left( x_i - r_i \right)^2}

  * ``'pearsonr'``: `Unweighted Pearson correlation coefficient`_.

  .. math::

    r = \frac{
      \sum_{i=1}^N
      \left( x_i - \bar{x} \right) \left( r_i - \bar{r} \right)
    }{
      \sqrt{\sum_{i=1}^N \left( x_i - \bar{x} \right)^2}
      \sqrt{\sum_{i=1}^N \left( r_i - \bar{r} \right)^2}
    }

  * ``'weighted_pearsonr'``: `Weighted Pearson correlation coefficient`_.

  .. math::

    r = \frac{
      \sum_{i=1}^N
      w_i \left( x_i - \bar{x} \right) \left( r_i - \bar{r} \right)
    }{
      \sqrt{\sum_{i=1}^N w_i \left( x_i - \bar{x} \right)^2}
      \sqrt{\sum_{i=1}^N w_i \left( r_i - \bar{r} \right)^2}
    }


  * ``'emd'``: `Unweighted Earth mover's distance`_ (EMD).
    The EMD is also known as first Wasserstein metric `W`\ :sub:`1`, which is a
    metric that measures distance between two probability distributions.
    For this, discrete probability distributions of the input data are created
    through binning, which are then used as input for the Wasserstein metric.
    The metric is also known as `Earth mover's distance` since, intuitively, it
    can be seen as the minimum "cost" of turning one pile of earth into another
    one (pile of earth = probability distribution).
    This is also known as `optimal transport` problem.
    Formally, this can be described with a joint probability distribution (or
    `optimal transport matrix`) γ (whose marginals are the input distributions)
    that minimizes the "transportation cost":

  .. math::

    W_1 = \min_{\gamma \in \mathbb{R}^{n \times n}_{+}} \sum_{i,j}^{n}
    \gamma_{ij} \lvert X_i - R_i \rvert \\
    \textrm{with} ~~ \gamma 1 = p_X(X);~ \gamma^T 1 = p_R(R)

  * ``'weighted_emd'``: `Weighted Earth mover's distance`_.
    Similar to the unweighted EMD (see above), but here weights are considered
    when calculating the probability distributions (i.e., instead of 1, each
    element provides a weight in the bin count; see also `weights`
    argument of :func:`numpy.histogram`).

  Here, `x`\ :sub:`i` and `r`\ :sub:`i` are samples of a variable of interest
  and a corresponding reference, respectively (a bar over a variable denotes
  its arithmetic/weighted mean [the latter for weighted metrics]).
  Capital letters (`X`\ :sub:`i` and `R`\ :sub:`i`) refer to bin centers of a
  discrete probability distribution with values `p`\ :sub:`X`\ (`X`\ :sub:`i`)
  or `p`\ :sub:`R`\ (`R`\ :sub:`i`) and a number of bins `n` (see the argument
  ``n_bins`` below) that has been derived for the variables `x` and `r` through
  binning.
  `w`\ :sub:`i` are weights that sum to one (see note below) and `N` is the
  total number of samples.

  .. note::
    Metrics starting with `weighted_` will calculate weighted distance metrics
    if possible.
    Currently, the following `coords` (or any combinations that include them)
    will trigger weighting: `time` (will use lengths of time intervals as
    weights) and `latitude` (will use cell area weights).
    Time weights are always calculated from the input data.
    Area weights can be given as supplementary variables to the recipe
    (`areacella` or `areacello`, see :ref:`supplementary_variables`) or
    calculated from the input data (this only works for regular grids).
    By default, **NO** supplementary variables will be used; they need to be
    explicitly requested in the recipe.
* ``coords`` (:obj:`list` of :obj:`str`, default: ``None``): Coordinates over
  which the distance metric is calculated.
  If ``None``, calculate the metric over all coordinates, which results in a
  scalar cube.
* ``keep_reference_dataset`` (:obj:`bool`, default: ``True``): If ``True``,
  also calculate the distance of the reference dataset with itself.
  If ``False``, drop the reference dataset.
* ``exclude`` (:obj:`list` of :obj:`str`): Exclude specific datasets from
  this preprocessor.
  Note that this option is only available in the recipe, not when using
  :func:`esmvalcore.preprocessor.distance_metric` directly (e.g., in another
  python script).
  If the reference dataset has been excluded, an error is raised.
* Other parameters are directly used for the metric calculation.
  The following keyword arguments are supported:

  * `rmse` and `weighted_rmse`: none.
  * `pearsonr` and `weighted_pearsonr`: ``mdtol``, ``common_mask`` (all keyword
    arguments are passed to :func:`iris.analysis.stats.pearsonr`, see that link
    for more details on these arguments).
    Note: in contrast to :func:`~iris.analysis.stats.pearsonr`,
    ``common_mask=True`` by default.
  * `emd` and `weighted_emd`: ``n_bins`` = number of bins used to create
    discrete probability distribution of data before calculating the EMD
    (:obj:`int`, default: 100).

Example:

.. code-block:: yaml

    preprocessors:
      preproc_pearsonr:
        distance_metric:
          metric: weighted_pearsonr
          coords: [latitude, longitude]
          keep_reference_dataset: true
          exclude: [CanESM2]
          common_mask: true

See also :func:`esmvalcore.preprocessor.distance_metric`.

.. _Unweighted root mean square error: https://en.wikipedia.org/wiki/
  Root-mean-square_deviation
.. _Weighted root mean square error: https://en.wikipedia.org/wiki/
  Root-mean-square_deviation
.. _Unweighted Pearson correlation coefficient: https://en.wikipedia.org/
  wiki/Pearson_correlation_coefficient
.. _Weighted Pearson correlation coefficient: https://en.wikipedia.org/
  wiki/Pearson_correlation_coefficient
.. _Unweighted Earth mover's distance: https://pythonot.github.io/
  quickstart.html#computing-wasserstein-distance
.. _Weighted Earth mover's distance: https://pythonot.github.io/
  quickstart.html#computing-wasserstein-distance


.. _Memory use:

Information on maximum memory required
======================================
In the most general case, we can set upper limits on the maximum memory the
analysis will require:


``Ms = (R + N) x F_eff - F_eff`` - when no multi-model analysis is performed;

``Mm = (2R + N) x F_eff - 2F_eff`` - when multi-model analysis is performed;

where

* ``Ms``: maximum memory for non-multimodel module
* ``Mm``: maximum memory for multi-model module
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
multi-model analysis could be estimated by multiplying the number of datasets by
the average file size of all the datasets; this memory intake is high but also
assumes that all data is fully realized in memory; this aspect will gradually
change and the amount of realized data will decrease with the increase of
``dask`` use.

.. _Other:

Other
=====

Miscellaneous functions that do not belong to any of the other categories.

Clip
----

This function clips data values to a certain minimum, maximum or range. The function takes two
arguments:

* ``minimum``: Lower bound of range. Default: ``None``
* ``maximum``: Upper bound of range. Default: ``None``

The example below shows how to set all values below zero to zero.


.. code-block:: yaml

    preprocessors:
      clip:
        minimum: 0
        maximum: null

.. _histogram:

``histogram``
-------------------

This function calculates histograms.

The ``histogram`` preprocessor supports the following arguments in the
recipe:

* ``coords`` (:obj:`list` of :obj:`str`, default: ``None``): Coordinates over
  which the histogram is calculated.
  If ``None``, calculate the histogram over all coordinates.
  The shape of the output cube will be `(x1, x2, ..., n_bins)`, where `xi` are
  the dimensions of the input cube not appearing in `coords` and `n_bins` is
  the number of bins.
* ``bins`` (:obj:`int` or sequence of :obj:`float`, default: 10): If `bins` is
  an :obj:`int`, it defines the number of equal-width bins in the given
  `bin_range`.
  If `bins` is a sequence, it defines a monotonically increasing array of bin
  edges, including the rightmost edge, allowing for non-uniform bin widths.
* ``bin_range`` (:obj:`tuple` of :obj:`float` or ``None``, default: ``None``):
  The lower and upper range of the bins.
  If ``None``, `bin_range` is simply (``cube.core_data().min(),
  cube.core_data().max()``).
  Values outside the range are ignored.
  The first element of the range must be less than or equal to the second.
  `bin_range` affects the automatic bin computation as well if `bins` is an
  :obj:`int` (see description for `bins` above).
* ``weights`` (array-like, :obj:`bool`, or ``None``, default: ``None``):
  Weights for the histogram calculation.
  Each value in the input data only contributes its associated weight towards
  the bin count (instead of 1).
  Weights are normalized before entering the calculation if `normalization` is
  ``'integral'`` or ``'sum'``.
  Can be an array of the same shape as the input data, ``False`` or ``None``
  (no weighting), or ``True``.
  In the latter case, weighting will depend on `coords`, and the following
  coordinates will trigger weighting: `time` (will use lengths of time
  intervals as weights) and `latitude` (will use cell area weights).
  Time weights are always calculated from the input data.
  Area weights can be given as supplementary variables in the recipe
  (`areacella` or `areacello`, see :ref:`supplementary_variables`) or
  calculated from the input data (this only works for regular grids).
  By default, **NO** supplementary variables will be used; they need to be
  explicitly requested in the recipe.
* ``normalization`` (``None``, ``'sum'``, or ``'integral'``, default:
  ``None``): If ``None``, the result will contain the number of samples in each
  bin.
  If ``'integral'``, the result is the value of the probability `density`
  function at the bin, normalized such that the integral over the range is 1.
  If ``'sum'``, the result is the value of the probability `mass` function at
  the bin, normalized such that the sum over the whole range is 1.
  Normalization will be applied across `coords`, not the entire cube.

Example:

.. code-block:: yaml

    preprocessors:
      preproc_histogram:
        histogram:
          coords: [latitude, longitude]
          bins: 12
          bin_range: [100.0, 150.0]
          weights: true
          normalization: sum

See also :func:`esmvalcore.preprocessor.histogram`.
