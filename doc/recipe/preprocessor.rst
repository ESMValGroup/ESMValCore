.. _preprocessor:

************
Preprocessor
************

In this section, each of the preprocessor modules is described,
roughly following the default order in which preprocessor functions are applied:

* :ref:`Variable derivation`
* :ref:`CMOR check and dataset-specific fixes`
* :ref:`Fx variables as cell measures or ancillary variables`
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
* :ref:`Bias`
* :ref:`Other`

See :ref:`preprocessor_functions` for implementation details and the exact default order.

Overview
========

..
   ESMValTool is a modular ``Python 3.8+`` software package possessing capabilities
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

Data preprocessed by ESMValTool is automatically checked against its
cmor definition. To reduce the impact of this check while maintaining
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

.. _Fx variables as cell measures or ancillary variables:

Fx variables as cell measures or ancillary variables
====================================================
The following preprocessors may require the use of ``fx_variables`` to be able
to perform the computations:

============================================================== =====================
Preprocessor                                                   Default fx variables
============================================================== =====================
:ref:`area_statistics<area_statistics>`                        ``areacella``, ``areacello``
:ref:`mask_landsea<land/sea/ice masking>`                      ``sftlf``, ``sftof``
:ref:`mask_landseaice<ice masking>`                            ``sftgif``
:ref:`volume_statistics<volume_statistics>`                    ``volcello``
:ref:`weighting_landsea_fraction<land/sea fraction weighting>` ``sftlf``, ``sftof``
============================================================== =====================

If the option ``fx_variables`` is not explicitly specified for these
preprocessors, the default fx variables in the second column are automatically
used. If given, the ``fx_variables`` argument specifies the fx variables that
the user wishes to input to the corresponding preprocessor function. The user
may specify these by simply adding the names of the variables, e.g.,

.. code-block:: yaml

    fx_variables:
      areacello:
      volcello:

or by additionally specifying further keys that are used to define the fx
datasets, e.g.,

.. code-block:: yaml

    fx_variables:
      areacello:
        mip: Ofx
        exp: piControl
      volcello:
        mip: Omon

This might be useful to select fx files from a specific ``mip`` table or from a
specific ``exp`` in case not all experiments provide the fx variable.

Alternatively, the ``fx_variables`` argument can also be specified as a list:

.. code-block:: yaml

    fx_variables: ['areacello', 'volcello']

or as a list of dictionaries:

.. code-block:: yaml

    fx_variables: [{'short_name': 'areacello', 'mip': 'Ofx', 'exp': 'piControl'}, {'short_name': 'volcello', 'mip': 'Omon'}]

The recipe parser will automatically find the data files that are associated
with these variables and pass them to the function for loading and processing.

If ``mip`` is not given, ESMValTool will search for the fx variable in all
available tables of the specified project.

.. warning::
   Some fx variables exist in more than one table (e.g., ``volcello`` exists in
   CMIP6's ``Odec``, ``Ofx``, ``Omon``, and ``Oyr`` tables; ``sftgif`` exists
   in CMIP6's ``fx``, ``IyrAnt`` and ``IyrGre``, and ``LImon`` tables). If (for
   a given dataset) fx files are found in more than one table, ``mip`` needs to
   be specified, otherwise an error is raised.

.. note::
   To explicitly **not** use any fx variables in a preprocessor, use
   ``fx_variables: null``.  While some of the preprocessors mentioned above do
   work without fx variables (e.g., ``area_statistics`` or ``mask_landsea``
   with datasets that have regular latitude/longitude grids), using this option
   is **not** recommended.

Internally, the required ``fx_variables`` are automatically loaded by the
preprocessor step ``add_fx_variables`` which also checks them against CMOR
standards and adds them either as ``cell_measure`` (see `CF conventions on cell
measures
<https://cfconventions.org/cf-conventions/cf-conventions.html#cell-measures>`_
and :class:`iris.coords.CellMeasure`) or ``ancillary_variable`` (see `CF
conventions on ancillary variables
<https://cfconventions.org/cf-conventions/cf-conventions.html#ancillary-data>`_
and :class:`iris.coords.AncillaryVariable`) inside the cube data. This ensures
that the defined preprocessor chain is applied to both ``variables`` and
``fx_variables``.

Note that when calling steps that require ``fx_variables`` inside diagnostic
scripts, the variables are expected to contain the required ``cell_measures`` or
``ancillary_variables``. If missing, they can be added using the following functions:

.. code-block::

    from esmvalcore.preprocessor import (add_cell_measure, add_ancillary_variable)

    cube_with_area_measure = add_cell_measure(cube, area_cube, 'area')

    cube_with_volume_measure = add_cell_measure(cube, volume_cube, 'volume)

    cube_with_ancillary_sftlf = add_ancillary_variable(cube, sftlf_cube)

    cube_with_ancillary_sftgif = add_ancillary_variable(cube, sftgif_cube)

  Details on the arguments needed for each step can be found in the following sections.

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

Optionally you can specify your own custom fx variable to be used in cases when
e.g. a certain experiment is preferred for fx data retrieval:

.. code-block:: yaml

    preprocessors:
      preproc_weighting:
        weighting_landsea_fraction:
          area_type: land
          exclude: ['CanESM2', 'reference_dataset']
          fx_variables:
            sftlf:
              exp: piControl
            sftof:
              exp: piControl

or alternatively:

.. code-block:: yaml

    preprocessors:
      preproc_weighting:
        weighting_landsea_fraction:
          area_type: land
          exclude: ['CanESM2', 'reference_dataset']
          fx_variables: [
            {'short_name': 'sftlf', 'exp': 'piControl'},
            {'short_name': 'sftof', 'exp': 'piControl'}
            ]

More details on the argument ``fx_variables`` and its default values are given
in :ref:`Fx variables as cell measures or ancillary variables`.

See also :func:`esmvalcore.preprocessor.weighting_landsea_fraction`.


.. _masking:

Masking
=======

Introduction to masking
-----------------------

Certain metrics and diagnostics need to be computed and performed on specific
domains on the globe. The ESMValTool preprocessor supports filtering
the input data on continents, oceans/seas and ice. This is achieved by masking
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

Optionally you can specify your own custom fx variable to be used in cases when e.g. a certain
experiment is preferred for fx data retrieval. Note that it is possible to specify as many tags
for the fx variable as required:


.. code-block:: yaml

    preprocessors:
      landmask:
        mask_landsea:
          mask_out: sea
          fx_variables:
            sftlf:
              exp: piControl
            sftof:
              exp: piControl
              ensemble: r2i1p1f1

or alternatively:

.. code-block:: yaml

    preprocessors:
      landmask:
        mask_landsea:
          mask_out: sea
          fx_variables: [
            {'short_name': 'sftlf', 'exp': 'piControl'},
            {'short_name': 'sftof', 'exp': 'piControl', 'ensemble': 'r2i1p1f1'}
            ]

More details on the argument ``fx_variables`` and its default values are given
in :ref:`Fx variables as cell measures or ancillary variables`.

If the corresponding fx file is not found (which is
the case for some models and almost all observational datasets), the
preprocessor attempts to mask the data using Natural Earth mask files (that are
vectorized rasters). As mentioned above, the spatial resolution of the the
Natural Earth masks are much higher than any typical global model (10m for
land and glaciated areas and 50m for ocean masks).

See also :func:`esmvalcore.preprocessor.mask_landsea`.

.. _ice masking:

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

Optionally you can specify your own custom fx variable to be used in cases when
e.g. a certain experiment is preferred for fx data retrieval:


.. code-block:: yaml

    preprocessors:
      landseaicemask:
        mask_landseaice:
          mask_out: sea
          fx_variables:
            sftgif:
              exp: piControl

or alternatively:

.. code-block:: yaml

    preprocessors:
      landseaicemask:
        mask_landseaice:
          mask_out: sea
          fx_variables: [{'short_name': 'sftgif', 'exp': 'piControl'}]

More details on the argument ``fx_variables`` and its default values are given
in :ref:`Fx variables as cell measures or ancillary variables`.

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
This is handled in ESMValTool via the missing values masks: two types of such
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

The underlying regridding mechanism in ESMValTool uses
:obj:`iris.cube.Cube.regrid`
from Iris.

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

ESMValTool has a number of built-in regridding schemes, which are presented in
:ref:`built-in regridding schemes`. Additionally, it is also possible to use
third party regridding schemes designed for use with :doc:`Iris
<iris:index>`. This is explained in :ref:`generic regridding schemes`.

.. _built-in regridding schemes:

Built-in regridding schemes
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The schemes used for the interpolation and extrapolation operations needed by
the horizontal regridding functionality directly map to their corresponding
implementations in :mod:`iris`:

* ``linear``: Linear interpolation without extrapolation, i.e., extrapolation
  points will be masked even if the source data is not a masked array (uses
  ``Linear(extrapolation_mode='mask')``, see :obj:`iris.analysis.Linear`).
* ``linear_extrapolate``: Linear interpolation with extrapolation, i.e.,
  extrapolation points will be calculated by extending the gradient of the
  closest two points (uses ``Linear(extrapolation_mode='extrapolate')``, see
  :obj:`iris.analysis.Linear`).
* ``nearest``: Nearest-neighbour interpolation without extrapolation, i.e.,
  extrapolation points will be masked even if the source data is not a masked
  array (uses ``Nearest(extrapolation_mode='mask')``, see
  :obj:`iris.analysis.Nearest`).
* ``area_weighted``: Area-weighted regridding (uses ``AreaWeighted()``, see
  :obj:`iris.analysis.AreaWeighted`).
* ``unstructured_nearest``: Nearest-neighbour interpolation for unstructured
  grids (uses ``UnstructuredNearest()``, see
  :obj:`iris.analysis.UnstructuredNearest`).

See also :func:`esmvalcore.preprocessor.regrid`

.. note::

   Controlling the extrapolation mode allows us to avoid situations where
   extrapolating values makes little physical sense (e.g. extrapolating beyond
   the last data point).

.. note::

   The regridding mechanism is (at the moment) done with fully realized data in
   memory, so depending on how fine the target grid is, it may use a rather
   large amount of memory. Empirically target grids of up to ``0.5x0.5``
   degrees should not produce any memory-related issues, but be advised that
   for resolutions of ``< 0.5`` degrees the regridding becomes very slow and
   will use a lot of memory.

.. _generic regridding schemes:

Generic regridding schemes
~~~~~~~~~~~~~~~~~~~~~~~~~~

:ref:`Iris' regridding <iris:interpolation_and_regridding>` is based around the
flexible use of so-called regridding schemes. These are classes that know how
to transform a source cube with a given grid into the grid defined by a given
target cube. Iris itself provides a number of useful schemes, but they are
largely limited to work with simple, regular grids. Other schemes can be
provided independently. This is interesting when special regridding-needs arise
or when more involved grids and meshes need to be considered. Furthermore, it
may be desirable to have finer control over the parameters of the scheme than
is afforded by the built-in schemes described above.

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

The value of the ``reference`` key has two parts that are separated by a
``:`` with no surrounding spaces. The first part is an importable Python
module, the second refers to the scheme, i.e. some callable that will be called
with the remaining entries of the ``scheme`` dictionary passed as keyword
arguments.

One package that aims to capitalize on the :ref:`support for unstructured
meshes introduced in Iris 3.2 <iris:ugrid>` is
:doc:`iris-esmf-regrid:index`. It aims to provide lazy regridding for
structured regular and irregular grids, as well as unstructured meshes. An
example of its usage in an ESMValTool preprocessor is:

.. code-block:: yaml

    preprocessors:
      regrid_preprocessor:
        regrid:
          target_grid: 2.5x2.5
          scheme:
            reference: esmf_regrid.schemes:ESMFAreaWeighted
            mdtol: 0.7

.. TODO: Remove the following warning once things have settled a bit.
.. warning::
   Just as the mesh support in Iris itself, this new regridding package is
   still considered experimental.

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
is more straigtforward and can benefit from lazy evaluation and more efficient
computation.

The preprocessor takes a list of statistics as input:

.. code-block:: yaml

    preprocessors:
      example_preprocessor:
        ensemble_statistics:
          statistics: [mean, median]

This preprocessor function exposes the iris analysis package, and works with all
(capitalized) statistics from the :mod:`iris.analysis` package
that can be executed without additional arguments (e.g. percentiles are not
supported because it requires additional keywords: percentile.).

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
when using a multi-model ensemble. ESMValTool has the capability of computing a
number of multi-model statistical measures: using the preprocessor module
``multi_model_statistics`` will enable the user to ask for either a multi-model
``mean``, ``median``, ``max``, ``min``, ``std_dev``, and / or ``pXX.YY`` with a set
of argument parameters passed to ``multi_model_statistics``. Percentiles can be
specified like ``p1.5`` or ``p95``. The decimal point will be replaced by a dash
in the output file.

Restrictive computation is also available by excluding  any set of models that
the user will not want to include in the statistics (by setting ``exclude:
[excluded models list]`` argument). The implementation has a few restrictions
that apply to the input data: model datasets must have consistent shapes, apart
from the time dimension; and cubes with more than four dimensions (time,
vertical axis, two horizontal axes) are not supported.

Input datasets may have different time coordinates. Statistics can be computed
across overlapping times only (``span: overlap``) or across the full time span
of the combined models (``span: full``). The preprocessor sets a common time
coordinate on all datasets. As the number of days in a year may vary between
calendars, (sub-)daily data with different calendars are not supported.
The preprocessor saves both the input single model files as well as the multi-model
results. In case you do not want to keep the single model files, set the
parameter ``keep_input_datasets`` to ``false`` (default value is ``true``).

.. code-block:: yaml

    preprocessors:
      multi_model_save_input:
        multi_model_statistics:
          span: overlap
          statistics: [mean, median]
          exclude: [NCEP]
      multi_model_without_saving_input:
        multi_model_statistics:
          span: overlap
          statistics: [mean, median]
          exclude: [NCEP]
          keep_input_datasets: false

Input datasets may have different time coordinates. The multi-model statistics
preprocessor sets a common time coordinate on all datasets. As the number of
days in a year may vary between calendars, (sub-)daily data are not supported.

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
          exclude: [NCEP]

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

.. note::

   The multi-model array operations can be rather memory-intensive (since they
   are not performed lazily as yet). The Section on :ref:`Memory use` details
   the memory intake for different run scenarios, but as a thumb rule, for the
   multi-model preprocessor, the expected maximum memory intake could be
   approximated as the number of datasets multiplied by the average size in
   memory for one dataset.

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
* regrid_time_: Aligns the time axis of each dataset to have common time
  points and calendars.
* timeseries_filter_: Allows application of a filter to the time-series data.

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
    * every_n_hours: frequency to use to compute the statistics. Must be a divisor of
      24.

    * operator: operation to apply. Accepted values are 'mean',
      'median', 'std_dev', 'min', 'max' and 'sum'. Default is 'mean'

See also :func:`esmvalcore.preprocessor.daily_statistics`.

.. _daily_statistics:

``daily_statistics``
--------------------

This function produces statistics for each day in the dataset.

Parameters:
    * operator: operation to apply. Accepted values are 'mean',
      'median', 'std_dev', 'min', 'max', 'sum' and 'rms'. Default is 'mean'

See also :func:`esmvalcore.preprocessor.daily_statistics`.

.. _monthly_statistics:

``monthly_statistics``
----------------------

This function produces statistics for each month in the dataset.

Parameters:
    * operator: operation to apply. Accepted values are 'mean',
      'median', 'std_dev', 'min', 'max', 'sum' and 'rms'. Default is 'mean'

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
December and remove such biased initial datapoints.

Parameters:
    * operator: operation to apply. Accepted values are 'mean',
      'median', 'std_dev', 'min', 'max', 'sum' and 'rms'. Default is 'mean'

    * seasons: seasons to build statistics.
      Default is '[DJF, MAM, JJA, SON]'

See also :func:`esmvalcore.preprocessor.seasonal_statistics`.

.. _annual_statistics:

``annual_statistics``
---------------------

This function produces statistics for each year.

Parameters:
    * operator: operation to apply. Accepted values are 'mean',
      'median', 'std_dev', 'min', 'max', 'sum' and 'rms'. Default is 'mean'

See also :func:`esmvalcore.preprocessor.annual_statistics`.

.. _decadal_statistics:

``decadal_statistics``
----------------------

This function produces statistics for each decade.

Parameters:
    * operator: operation to apply. Accepted values are 'mean',
      'median', 'std_dev', 'min', 'max', 'sum' and 'rms'. Default is 'mean'

See also :func:`esmvalcore.preprocessor.decadal_statistics`.

.. _climate_statistics:

``climate_statistics``
----------------------

This function produces statistics for the whole dataset. It can produce scalars
(if the full period is chosen) or daily, monthly or seasonal statistics.

Parameters:
    * operator: operation to apply. Accepted values are 'mean', 'median',
      'std_dev', 'min', 'max', 'sum' and 'rms'. Default is 'mean'

    * period: define the granularity of the statistics: get values for the
      full period, for each month or day of year.
      Available periods: 'full', 'season', 'seasonal', 'monthly', 'month',
      'mon', 'daily', 'day'. Default is 'full'

    * seasons: if period 'seasonal' or 'season' allows to set custom seasons.
      Default is '[DJF, MAM, JJA, SON]'

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
anomalies from the full, seasonal, monthly and daily climatologies. Optionally
standardized anomalies can be calculated.

Parameters:
    * period: define the granularity of the climatology to use:
      full period, seasonal, monthly or daily.
      Available periods: 'full', 'season', 'seasonal', 'monthly', 'month',
      'mon', 'daily', 'day'. Default is 'full'
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

This function aligns the time points of each component dataset so that the Iris
cubes from different datasets can be subtracted. The operation makes the
datasets time points common; it also resets the time
bounds and auxiliary coordinates to reflect the artificially shifted time
points. Current implementation for monthly and daily data; the ``frequency`` is
set automatically from the variable CMOR table unless a custom ``frequency`` is
set manually by the user in recipe.

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
-------------------------

Extract a shape or a representative point for this shape from
the data.

Parameters:
  * ``shapefile``: path to the shapefile containing the geometry of the
    region to be extracted. If the file contains multiple shapes behaviour
    depends on the decomposed parameter. This path can be relative to
    ``auxiliary_data_dir`` defined in the :ref:`user configuration file`.
  * ``method``: the method to select the region, selecting either all points
	  contained by the shape or a single representative point. Choose either
	  'contains' or 'representative'. If not a single grid point is contained
	  in the shape, a representative point will be selected.
  * ``crop``: by default extract_region_ will be used to crop the data to a
	  minimal rectangular region containing the shape. Set to ``false`` to only
	  mask data outside the shape. Data on irregular grids will not be cropped.
  * ``decomposed``: by default ``false``, in this case the union of all the
    regions in the shape file is masked out. If ``true``, the regions in the
    shapefiles are masked out separately, generating an auxiliary dimension
    for the cube for this.
  * ``ids``: by default, ``[]``, in this case all the shapes in the file will
    be used. If a list of IDs is provided, only the shapes matching them will
    be used. The IDs are assigned from the ``name`` or ``id`` attributes (in
    that order of priority) if present in the file or from the reading order
    if otherwise not present. So, for example, if a file has both ```name``
    and ``id`` attributes, the ids will be assigned from ``name``. If the file
    only has the ``id`` attribute, it will be taken from it and if no ``name``
    nor ``id`` attributes are present, an integer id starting from 1 will be
    assigned automatically when reading the shapes. We discourage to rely on
    this last behaviour as we can not assure that the reading order will be the
    same in different platforms, so we encourage you to modify the file to add
    a proper id attribute. If the file has an id attribute with a name that is
    not supported, please open an issue so we can add support for it.

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

Parameters
  * ``cube``: the input dataset cube to extract a point from.
  * ``location``: the reference location. Examples: 'mount everest',
    'romania', 'new york, usa'. Raises ValueError if none supplied.
  * ``scheme`` : interpolation scheme. ``'linear'`` or ``'nearest'``.
    There is no default, raises ValueError if none supplied.

See also :func:`esmvalcore.preprocessor.extract_location`.


``zonal_statistics``
--------------------

The function calculates the zonal statistics by applying an operator
along the longitude coordinate. This function takes one argument:

* ``operator``: Which operation to apply: mean, std_dev, median, min, max, sum or rms.

See also :func:`esmvalcore.preprocessor.zonal_means`.


``meridional_statistics``
-------------------------

The function calculates the meridional statistics by applying an
operator along the latitude coordinate. This function takes one
argument:

* ``operator``: Which operation to apply: mean, std_dev, median, min, max, sum or rms.

See also :func:`esmvalcore.preprocessor.meridional_means`.


.. _area_statistics:

``area_statistics``
-------------------

This function calculates the average value over a region - weighted by the cell
areas of the region. This function takes the argument, ``operator``: the name
of the operation to apply.

This function can be used to apply several different operations in the
horizontal plane: mean, standard deviation, median, variance, minimum, maximum and root mean square.

Note that this function is applied over the entire dataset. If only a specific
region, depth layer or time period is required, then those regions need to be
removed using other preprocessor operations in advance.

The optional ``fx_variables`` argument specifies the fx variables that the user
wishes to input to the function. More details on this are given in :ref:`Fx
variables as cell measures or ancillary variables`.

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

Extract a specific range in the `z`-direction from a cube.  This function
takes two arguments, a minimum and a maximum (``z_min`` and ``z_max``,
respectively) in the `z`-direction.

Note that this requires the requested `z`-coordinate range to be the same sign
as the Iris cube. That is, if the cube has `z`-coordinate as negative, then
``z_min`` and ``z_max`` need to be negative numbers.

See also :func:`esmvalcore.preprocessor.extract_volume`.


.. _volume_statistics:

``volume_statistics``
---------------------

This function calculates the volume-weighted average across three dimensions,
but maintains the time dimension.

This function takes the argument: ``operator``, which defines the operation to
apply over the volume.

No depth coordinate is required as this is determined by Iris. This function
works best when the ``fx_variables`` provide the cell volume. The optional
``fx_variables`` argument specifies the fx variables that the user wishes to
input to the function. More details on this are given in :ref:`Fx variables as
cell measures or ancillary variables`.

See also :func:`esmvalcore.preprocessor.volume_statistics`.


``axis_statistics``
---------------------

This function operates over a given axis, and removes it from the
output cube.

Takes arguments:
  * axis: direction over which the statistics will be performed.
    Possible values for the axis are 'x', 'y', 'z', 't'.
  * operator: defines the operation to apply over the axis.
    Available operator are 'mean', 'median', 'std_dev', 'sum', 'variance',
    'min', 'max', 'rms'.

.. note::
   The coordinate associated to the axis over which the operation will
   be performed must be one-dimensional, as multidimensional coordinates
   are not supported in this preprocessor.

See also :func:`esmvalcore.preprocessor.axis_statistics`.


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

ESMValTool also supports detrending along any dimension using
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

* ``coordinate``: coordinate over which the rolling-window statistics is 
  calculated.

* ``operator``: operation to apply. Accepted values are 'mean', 'median',
  'std_dev', 'min', 'max' and 'sum'.

* ``window_length``: size of the rolling window to use (number of points).

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


.. _bias:

Bias
====

The bias module contains the following preprocessor functions:

* ``bias``: Calculate absolute or relative biases with respect to a reference
  dataset

``bias``
--------

This function calculates biases with respect to a given reference dataset. For
this, exactly one input dataset needs to be declared as ``reference_for_bias:
true`` in the recipe, e.g.,

.. code-block:: yaml

  datasets:
    - {dataset: CanESM5, project: CMIP6, ensemble: r1i1p1f1, grid: gn}
    - {dataset: CESM2,   project: CMIP6, ensemble: r1i1p1f1, grid: gn}
    - {dataset: MIROC6,  project: CMIP6, ensemble: r1i1p1f1, grid: gn}
    - {dataset: ERA-Interim, project: OBS6, tier: 3, type: reanaly, version: 1,
       reference_for_bias: true}

In the example above, ERA-Interim is used as reference dataset for the bias
calculation. For this preprocessor, all input datasets need to have identical
dimensional coordinates. This can for example be ensured with the preprocessors
:func:`esmvalcore.preprocessor.regrid` and/or
:func:`esmvalcore.preprocessor.regrid_time`.

The ``bias`` preprocessor supports 4 optional arguments:

   * ``bias_type`` (:obj:`str`, default: ``'absolute'``): Bias type that is
     calculated. Can be ``'absolute'`` (i.e., calculate bias for dataset
     :math:`X` and reference :math:`R` as :math:`X - R`) or ``relative`` (i.e,
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
