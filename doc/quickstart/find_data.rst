.. _findingdata:

**********
Input data
**********

Overview
========
Data discovery and retrieval is the first step in any evaluation process;
ESMValCore uses a `semi-automated` data finding mechanism with inputs from both
the configuration and the recipe file: this means that the user will
have to provide the tool with a set of parameters related to the data needed
and once these parameters have been provided, the tool will automatically find
the right data. We will detail below the data finding and retrieval process and
the input the user needs to specify, giving examples on how to use the data
finding routine under different scenarios.

Data types
==========

.. _CMOR-DRS:

CMIP data
---------
CMIP data is widely available via the Earth System Grid Federation
(`ESGF <https://esgf.llnl.gov/>`_) and is accessible to users either
via automatic download by ``esmvaltool`` or through the ESGF data nodes hosted
by large computing facilities (like CEDA-Jasmin, DKRZ, etc). This data
adheres to, among other standards, the DRS and Controlled Vocabulary
standard for naming files and structured paths; the `DRS
<https://www.ecmwf.int/sites/default/files/elibrary/2014/13713-data-reference-syntax-governing-standards-within-climate-research-data-archived-esgf.pdf>`_
ensures that files and paths to them are named according to a
standardized convention. Examples of this convention, also used by
ESMValCore for file discovery and data retrieval, include:

* CMIP6 file: ``{variable_short_name}_{mip}_{dataset_name}_{experiment}_{ensemble}_{grid}_{start-date}-{end-date}.nc``
* CMIP5 file: ``{variable_short_name}_{mip}_{dataset_name}_{experiment}_{ensemble}_{start-date}-{end-date}.nc``
* OBS file: ``{project}_{dataset_name}_{type}_{version}_{mip}_{short_name}_{start-date}-{end-date}.nc``

Similar standards exist for the standard paths (input directories); for the
ESGF data nodes, these paths differ slightly, for example:

* CMIP6 path for BADC: ``ROOT-BADC/{institute}/{dataset_name}/{experiment}/{ensemble}/{mip}/
  {variable_short_name}/{grid}``;
* CMIP6 path for ETHZ: ``ROOT-ETHZ/{experiment}/{mip}/{variable_short_name}/{dataset_name}/{ensemble}/{grid}``

From the ESMValCore user perspective the number of data input parameters is
optimized to allow for ease of use. We detail this procedure in the next
section.

Observational data
------------------
Part of observational data is retrieved in the same manner as CMIP data, for example
using the ``OBS`` root path set to:

  .. code-block:: yaml

    OBS: /gws/nopw/j04/esmeval/obsdata-v2

and the dataset:

  .. code-block:: yaml

    - {dataset: ERA-Interim, project: OBS6, type: reanaly, version: 1, start_year: 2014, end_year: 2015, tier: 3}

in ``recipe.yml`` in ``datasets`` or ``additional_datasets``, the rules set in
CMOR-DRS_ are used again and the file will be automatically found:

.. code-block::

  /gws/nopw/j04/esmeval/obsdata-v2/Tier3/ERA-Interim/OBS_ERA-Interim_reanaly_1_Amon_ta_201401-201412.nc

Observational datasets CMORized by ESMValTool are organized in Tiers depending on
their level of public availability.

.. _read_native_datasets:

Datasets in native format
-------------------------

Some datasets are supported in their native format (i.e., the data is not
formatted according to a CMIP data request) through the ``native6`` project
(mostly native reanalysis/observational datasets) or through a dedicated
project, e.g., ``ICON`` (mostly native models).
A detailed description of how to include new native datasets is given
:ref:`here <add_new_fix_native_datasets>`.

.. hint::

   When using native datasets, it might be helpful to specify a custom location
   for the :ref:`custom_cmor_tables`.
   This allows reading arbitrary variables from native datasets.
   Note that this requires the option ``cmor_strict: false`` in the
   :ref:`project configuration <configure_native_models>` used for the native
   model output.

.. _read_native_obs:

Supported native reanalysis/observational datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following native reanalysis/observational datasets are supported under the
``native6`` project.
To use these datasets, put the files containing the data in the directory that
you have :ref:`configured <config-data-sources>` for the ``rootpath`` of the
``native6`` project, in a subdirectory called
``Tier{tier}/{dataset}/{version}/{frequency}/{short_name}``, assuming you are
using the default DRS for ``native6`` as defined in the file:

.. literalinclude:: ../configurations/data-local-esmvaltool.yml
    :language: yaml
    :caption: Contents of ``data-local-esmvaltool.yml``
    :start-after: # Read native6, OBS6, and OBS data from the filesystem on a personal computer.
    :end-before: # Data that has been CMORized by ESMValTool according to the CMIP6 standard.

Replace the items in curly braces by the values used in the variable/dataset
definition in the :ref:`recipe <recipe_overview>`.

.. _read_native_era5_nc:

ERA5 (in netCDF format downloaded from the CDS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ERA5 data can be downloaded from the Copernicus Climate Data Store (CDS) using
the convenient tool `era5cli <https://era5cli.readthedocs.io>`__.
For example for monthly data, place the files in the
``/Tier3/ERA5/version/mon/pr`` subdirectory of your ``rootpath`` that you have
configured for the ``native6`` project (assuming you are using the default DRS
for ``native6`` described :ref:`above <read_native_obs>`).

- Supported variables: ``cl``, ``clt``, ``evspsbl``, ``evspsblpot``, ``mrro``,
  ``pr``, ``prsn``, ``ps``, ``psl``, ``ptype``, ``rls``, ``rlds``, ``rsds``,
  ``rsdt``, ``rss``, ``uas``, ``vas``, ``tas``, ``tasmax``, ``tasmin``,
  ``tdps``, ``ts``, ``tsn`` (``E1hr``/``Amon``), ``orog`` (``fx``).
- Tier: 3

.. note:: According to the description of Evapotranspiration and potential Evapotranspiration on the Copernicus page
  (https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels-monthly-means?tab=overview):
  "The ECMWF Integrated Forecasting System (IFS) convention is that downward fluxes are positive.
  Therefore, negative values indicate evaporation and positive values indicate condensation."

  In the CMOR table, these fluxes are defined as positive, if they go from the surface into the atmosphere:
  "Evaporation at surface (also known as evapotranspiration): flux of water into the atmosphere due to conversion
  of both liquid and solid phases to vapor (from underlying surface and vegetation)."
  Therefore, the ERA5 (and ERA5-Land) CMORizer switches the signs of ``evspsbl`` and ``evspsblpot`` to be compatible with the CMOR standard used e.g. by the CMIP models.

.. _read_native_era5_grib:

ERA5 (in GRIB format available on DKRZ's Levante or downloaded from the CDS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ERA5 data in monthly, daily, and hourly resolution is `available on Levante
<https://docs.dkrz.de/doc/dataservices/finding_and_accessing_data/era_data/index.html#era-data>`__
in its native GRIB format.

.. note::
  ERA5 data in its native GRIB format can also be downloaded from the
  `Copernicus Climate Data Store (CDS)
  <https://cds.climate.copernicus.eu/datasets>`__.
  For example, hourly data on pressure levels is available `here
  <https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=download>`__.
  Reading self-downloaded ERA5 data in GRIB format is experimental and likely
  requires additional setup from the user like setting up the proper directory
  structure for the input files.

To read these data with ESMValCore, use the data definition for the ``native6``
project:

.. literalinclude:: ../configurations/data-hpc-dkrz.yml
    :language: yaml
    :caption: Contents of ``data-hpc-dkrz.yml``
    :start-at: # ERA5 data in GRIB format:
    :end-before: OBS6:

To use this configuration, run ``esmvaltool config copy data-hpc-dkrz.yml``.

The `naming conventions
<https://docs.dkrz.de/doc/dataservices/finding_and_accessing_data/era_data/index.html#file-and-directory-names>`__
for input directories and files for native ERA5 data in GRIB format on Levante
are

* input directories: ``{family}/{level}/{type}/{tres}/{grib_id}``
* input files: ``{family}{level}{typeid}_{tres}_*_{grib_id}.grb``

All of these facets have reasonable defaults preconfigured in the corresponding
:ref:`extra facets<config-extra-facets>` configuration file, which is available
here: :download:`extra_facets_native6.yml
</../esmvalcore/config/configurations/defaults/extra_facets_native6.yml>`.
If necessary, these facets can be overwritten in the recipe.

Thus, example dataset entries could look like this:

.. code-block:: yaml

  datasets:
    - {project: native6, dataset: ERA5, timerange: '2000/2001',
       short_name: tas, mip: Amon}
    - {project: native6, dataset: ERA5, timerange: '2000/2001',
       short_name: cl, mip: Amon, tres: 1H, frequency: 1hr}
    - {project: native6, dataset: ERA5, timerange: '2000/2001',
       short_name: ta, mip: Amon, type: fc, typeid: '12'}

The native ERA5 output in GRIB format is stored on a `reduced Gaussian grid
<https://confluence.ecmwf.int/display/CKB/ERA5:+data+documentation#ERA5:datadocumentation-SpatialgridSpatialGrid>`__.
By default, these data are regridded to a regular 0.25°x0.25° grid as
`recommended by the ECMWF
<https://confluence.ecmwf.int/display/CKB/ERA5%3A+What+is+the+spatial+reference#heading-Interpolation>`__
using bilinear interpolation.

To disable this, you can use the facet ``automatic_regrid: false`` in the
recipe:

.. code-block:: yaml

  datasets:
    - {project: native6, dataset: ERA5, timerange: '2000/2001',
       short_name: tas, mip: Amon, automatic_regrid: false}

- Supported variables: ``albsn``, ``cl``, ``cli``, ``clt``, ``clw``, ``hur``,
  ``hus``, ``o3``, ``prw``, ``ps``, ``psl``, ``rainmxrat27``, ``sftlf``,
  ``snd``, ``snowmxrat27``, ``ta``, ``tas``, ``tdps``, ``toz``, ``ts``, ``ua``,
  ``uas``, ``va``, ``vas``, ``wap``, ``zg``.

.. _read_native_mswep:

MSWEP
^^^^^

- Supported variables: ``pr``
- Supported frequencies: ``mon``, ``day``, ``3hr``.
- Tier: 3

For example for monthly data, place the files in the
``/Tier3/MSWEP/version/mon/pr`` subdirectory of your ``rootpath`` that you have
configured for the ``native6`` project (assuming you are using the default DRS
for ``native6`` described :ref:`above <read_native_obs>`).

.. note::
  For monthly data (``V220``), the data must be postfixed with the date, i.e. rename ``global_monthly_050deg.nc`` to ``global_monthly_050deg_197901-201710.nc``

For more info: http://www.gloh2o.org/

Data for the version ``V220`` can be downloaded from: https://hydrology.princeton.edu/data/hylkeb/MSWEP_V220/.

.. _read_native_models:

Supported native models
~~~~~~~~~~~~~~~~~~~~~~~

The following models are natively supported by ESMValCore.
In contrast to the native observational datasets listed above, they use
dedicated projects instead of the project ``native6``.

.. _read_cesm:

CESM
^^^^

ESMValCore is able to read native `CESM <https://www.cesm.ucar.edu/>`__ model
output.

.. warning::

   The support for native CESM output is still experimental.
   Currently, only one variable (`tas`) is fully supported. Other 2D variables
   might be supported by specifying appropriate facets in the recipe or extra
   facets files (see text below).
   3D variables (data that uses a vertical dimension) are not supported, yet.

The default naming conventions for input directories and files for CESM are

* input directories: 3 different types supported:
   * ``/`` (run directory)
   * ``{case}/{gcomp}/hist`` (short-term archiving)
   * ``{case}/{gcomp}/proc/{tdir}/{tperiod}`` (post-processed data)
* input files: ``{case}.{scomp}.{type}.{string}*nc``

as configured in:

.. literalinclude:: ../configurations/data-native-cesm.yml
    :language: yaml
    :caption: Contents of ``data-native-cesm.yml``

To use this configuration, run ``esmvaltool config copy data-native-cesm.yml`` and
adapt the ``rootpath`` to your system.
More information about CESM naming conventions are given `here
<https://www.cesm.ucar.edu/models/cesm2/naming_conventions.html>`__.

.. note::

   The ``{string}`` entry in the input file names above does not only
   correspond to the (optional) ``$string`` entry for `CESM model output files
   <https://www.cesm.ucar.edu/models/cesm2/naming_conventions.html#modelOutputFilenames>`__,
   but can also be used to read `post-processed files
   <https://www.cesm.ucar.edu/models/cesm2/naming_conventions.html#ppDataFilenames>`__.
   In the latter case, ``{string}`` corresponds to the combination
   ``$SSTRING.$TSTRING``.

Thus, example dataset entries could look like this:

.. code-block:: yaml

  datasets:
    - {project: CESM, dataset: CESM2, case: f.e21.FHIST_BGC.f09_f09_mg17.CMIP6-AMIP.001, type: h0, mip: Amon, short_name: tas, start_year: 2000, end_year: 2014}
    - {project: CESM, dataset: CESM2, case: f.e21.F1850_BGC.f09_f09_mg17.CFMIP-hadsst-piForcing.001, type: h0, gcomp: atm, scomp: cam, mip: Amon, short_name: tas, start_year: 2000, end_year: 2014}

Variable-specific defaults for the facet ``gcomp`` and ``scomp`` are given as
extra facets (see next paragraph) for some variables, but these can be
overwritten in the recipe.

Similar to any other fix, the CESM fix allows the use of :ref:`extra
facets<config-extra-facets>`.
The configuration file :download:`extra_facets_cesm.yml
</../esmvalcore/config/configurations/defaults/extra_facets_cesm.yml>` contains
the defaults.
Currently, this file only contains default facets for a single variable
(`tas`); for other variables, these entries need to be defined in the recipe.
Supported keys for extra facets are:

==================== ====================================== =================================
Key                  Description                            Default value if not specified
==================== ====================================== =================================
``gcomp``            Generic component-model name           No default (needs to be specified
                                                            as extra facets or in recipe if
                                                            default DRS is used)
``raw_name``         Variable name of the variable in the   CMOR variable name of the
                     raw input file                         corresponding variable
``raw_units``        Units of the variable in the raw       If specified, the value given by
                     input file                             the ``units`` attribute in the
                                                            raw input file; otherwise
                                                            ``unknown``
``scomp``            Specific component-model name          No default (needs to be specified
                                                            as extra facets or in recipe if
                                                            default DRS is used)
``string``           Short string which is used to further  ``''`` (empty string)
                     identify the history file type
                     (corresponds to ``$string`` or
                     ``$SSTRING.$TSTRING`` in the CESM file
                     name conventions; see note above)
``tdir``             Entry to distinguish time averages     ``''`` (empty string)
                     from time series from diagnostic plot
                     sets (only used for post-processed
                     data)
``tperiod``          Time period over which the data was    ``''`` (empty string)
                     processed (only used for
                     post-processed data)
==================== ====================================== =================================

.. _read_emac:

EMAC
^^^^

ESMValCore is able to read native `EMAC
<https://www.dlr.de/pa/en/desktopdefault.aspx/tabid-8859/15306_read-37415/>`_
model output.

The default naming conventions for input directories and files for EMAC are

* input directories: ``{exp}/{channel}``
* input files: ``{exp}*{channel}{postproc_flag}.nc``

as configured in:

.. literalinclude:: ../configurations/data-native-emac.yml
    :language: yaml
    :caption: Contents of ``data-native-emac.yml``

To use this configuration, run ``esmvaltool config copy data-native-emac.yml`` and
adapt the ``rootpath`` to your system.

Thus, example dataset entries could look like this:

.. code-block:: yaml

  datasets:
    - {project: EMAC, dataset: EMAC, exp: historical, mip: Amon, short_name: tas, start_year: 2000, end_year: 2014}
    - {project: EMAC, dataset: EMAC, exp: historical, mip: Omon, short_name: tos, postproc_flag: "-p-mm", start_year: 2000, end_year: 2014}
    - {project: EMAC, dataset: EMAC, exp: historical, mip: Amon, short_name: ta, raw_name: tm1_p39_cav, start_year: 2000, end_year: 2014}

Please note the duplication of the name ``EMAC`` in ``project`` and
``dataset``, which is necessary to comply with ESMValCore's data finding and
CMORizing functionalities.
A variable-specific default for the facet ``channel`` is given in the extra
facets (see next paragraph) for many variables, but this can be overwritten in
the recipe.

Similar to any other fix, the EMAC fix allows the use of :ref:`extra
facets<config-extra-facets>`.
The configuration file :download:`extra_facets_emac.yml
</../esmvalcore/config/configurations/defaults/extra_facets_emac.yml>` contains
the defaults.
For some variables, extra facets are necessary; otherwise ESMValCore cannot
read them properly.
Supported keys for extra facets are:

===================== ====================================== =================================
Key                   Description                            Default value if not specified
===================== ====================================== =================================
``channel``           Channel in which the desired variable  No default (needs to be specified
                      is stored                              as extra facets or in recipe if
                                                             default DRS is used)
``postproc_flag``     Postprocessing flag of the data        ``''`` (empty string)
``raw_name``          Variable name of the variable in the   CMOR variable name of the
                      raw input file                         corresponding variable
``raw_units``         Units of the variable in the raw       If specified, the value given by
                      input file                             the ``units`` attribute in the
                                                             raw input file; otherwise
                                                             ``unknown``
``reset_time_bounds`` Boolean if time bounds are deleted,    ``False``
                      and automatically recalculated by
                      iris
===================== ====================================== =================================

.. note::

   ``raw_name`` can be given as ``str`` or ``list``.
   The latter is used to support multiple different variables names in the
   input file.
   In this case, the prioritization is given by the order of the list; if
   possible, use the first entry, if this is not present, use the second, etc.
   This is particularly useful for files in which regular averages (``*_ave``)
   or conditional averages (``*_cav``) exist.

   For 3D variables defined on pressure levels, only the pressure levels
   defined by the CMOR table (e.g., for `Amon`'s `ta`: ``tm1_p19_cav`` and
   ``tm1_p19_ave``) are given as default extra facets.
   If other pressure levels are desired, e.g., ``tm1_p39_cav``, this has to be
   explicitly specified in the recipe using ``raw_name: tm1_p39_cav`` or
   ``raw_name: [tm1_p19_cav, tm1_p39_cav]``.

.. _read_icon:

ICON
^^^^

ESMValCore is able to read native `ICON <https://www.icon-model.org/>`__ model
output.

The default naming conventions for input directories and files for ICON are

* input directories: ``{exp}``, ``{exp}/outdata``, or ``{exp}/output``
* input files: ``{exp}_{var_type}*.nc``

as configured in:

.. literalinclude:: ../configurations/data-native-icon.yml
    :language: yaml
    :caption: Contents of ``data-native-icon.yml``

To use this configuration, run ``esmvaltool config copy data-native-icon.yml`` and
adapt the ``rootpath`` to your system.

Currently, two different versions of ICON are supported:

1. ICON-A, which is based on ECHAM physics (deprecated): select via ``dataset:
   ICON``.
2. ICON-XPP, which is based on NWP physics (preferred; model code can be
   downloaded from `DKRZ's GitLab <https://gitlab.dkrz.de/icon/icon-model>`__):
   select via ``dataset: ICON-XPP``.

Thus, example dataset entries could look like this:

.. code-block:: yaml

  datasets:
    - {project: ICON, dataset: ICON, exp: icon-2.6.1_atm_amip_R2B5_r1i1p1f1,
       mip: Amon, short_name: tas, timerange: 20010101/20020101}
    - {project: ICON, dataset: ICON-XPP, exp: historical, mip: Amon,
       short_name: ta, timerange: 20010101/20020101}

A variable-specific default for the facet ``var_type`` is given in the extra
facets (see below) for many variables, but this can be overwritten in the
recipe, for example:

.. code-block:: yaml

  datasets:
    - {project: ICON, dataset: ICON-XPP, exp: historical, mip: Amon,
       short_name: ta, var_type: atm_dyn_3d_ml, timerange: 20010101/20020101}

This is necessary if your ICON output is structured in one variable per file.
For example, if your output is stored in files called
``<exp>_<variable_name>_atm_2d_ml_YYYYMMDDThhmmss.nc``, use ``var_type:
<variable_name>_atm_2d_ml`` in the recipe for this variable.

Usually, ICON reports aggregated values at the end of the corresponding time
output intervals.
For example, for monthly output, ICON reports the month February as "1 March".
Thus, by default, ESMValCore shifts all time points back by 1/2 of the output
time interval so that the new time point corresponds to the center of the
interval.
This can be disabled by using ``shift_time: false`` in the recipe or the extra
facets (see below).
For point measurements (identified by ``cell_methods = "time: point"``), this
is always disabled.

.. warning::

   To get all desired time points, do **not** use ``start_year`` and
   ``end_year`` in the recipe, but rather ``timerange`` with at least 8 digits.
   For example, to get data for the years 2000 and 2001, use ``timerange:
   20000101/20020101`` instead of ``timerange: 2000/2001`` or ``start_year:
   2000``, ``end_year: 2001``.
   See :ref:`timerange_examples` for more information on the ``timerange``
   option.

Usually, ESMValCore will need the corresponding ICON grid file of your
simulation to work properly (examples: setting latitude/longitude coordinates
if these are not yet present, UGRIDization [see below], etc.).
This grid file can either be specified as absolute or relative (to the
:ref:`configuration option <config_options>` ``auxiliary_data_dir``) path with
the facet ``horizontal_grid`` in the recipe or as extra facet (see below), or
retrieved automatically from the `grid_file_uri` attribute of the input files.
In the latter case, ESMValCore first searches the input directories specified
for ICON for a grid file with that name, and if that was not successful, tries
to download the file and cache it.
The cached file is valid for 7 days.

ESMValCore can automatically make native ICON data `UGRID
<https://ugrid-conventions.github.io/ugrid-conventions/>`__-compliant when
loading the data.
The UGRID conventions provide a standardized format to store data on
unstructured grids, which is required by many software packages or tools to
work correctly and specifically by Iris to interpret the grid as a
:ref:`mesh <iris:ugrid>`.
An example is the horizontal regridding of native ICON data to a regular grid.
While the :ref:`built-in regridding schemes <default regridding schemes>`
`linear` and `nearest`  can handle unstructured grids (i.e., not UGRID-compliant) and meshes (i.e., UGRID-compliant),
the `area_weighted` scheme requires the input data in UGRID format.
This automatic UGRIDization is enabled by default, but can be switched off with
the facet ``ugrid: false`` in the recipe or as extra facet (see below).
This is useful for diagnostics that act on the native ICON grid and do not
support input data in UGRID format (yet).

For 3D ICON variables, ESMValCore tries to add the pressure level information
and/or altitude information to the preprocessed output files.
If the names of these variables differ from the default values, the facets
``pfull_var``, ``phalf_var``, ``zg_var``, and ``zghalf_var`` can be specified
in the recipe or as extra facets.
If neither of these variables are available in the input files, it is possible
to specify the location of files that include the corresponding altitude
information with the facets ``zg_file`` and/or ``zghalf_file`` in the recipe or
as extra facets.
The paths to these files can be specified absolute or relative (to the
:ref:`configuration option <config_options>` ``auxiliary_data_dir``).

.. hint::

   To use the :func:`~esmvalcore.preprocessor.extract_levels` preprocessor on
   native ICON data, you need to specify the name of the vertical coordinate
   (e.g., ``coordinate: air_pressure``) since native ICON output usually
   provides a 3D air pressure field instead of a simple 1D vertical coordinate.
   This also works if your files only contain altitude information (in this
   case, the US standard atmosphere is used to convert between altitude and
   pressure levels; see :ref:`Vertical interpolation` for details).
   Example:

   .. code-block:: yaml

    preprocessors:
      extract_500hPa_level_from_icon:
        extract_levels:
          levels: 50000
          scheme: linear
          coordinate: air_pressure

Similar to any other fix, the ICON fix allows the use of :ref:`extra
facets<config-extra-facets>`.
The configuration file :download:`extra_facets_icon.yml
</../esmvalcore/config/configurations/defaults/extra_facets_icon.yml>` contains
the defaults.
For some variables, extra facets are necessary; otherwise ESMValCore cannot
read them properly.
Supported keys for extra facets are:

=================== ================================ ===================================
Key                 Description                      Default value if not specified
=================== ================================ ===================================
``horizontal_grid`` Absolute or relative (to         If not given, use file attribute
                    ``auxiliary_data_dir``)          ``grid_file_uri`` to retrieve ICON
                    path to the ICON grid file       grid file (see details above)
``lat_var``         Variable name of the latitude    ``clat``
                    coordinate in the raw input
                    file/grid file
``lon_var``         Variable name of the longitude   ``clon``
                    coordinate in the raw input
                    file/grid file
``pfull_var``       Variable name of the pressure at ``pfull`` (``dataset: ICON``) or
                    full levels in the raw input     ``pres`` (``dataset: ICON-XPP``)
                    file
``phalf_var``       Variable name of the pressure at ``phalf``
                    half levels in the raw input
                    file
``raw_name``        Variable name of the             CMOR variable name of the
                    variable in the raw input        corresponding variable
                    file
``raw_units``       Units of the variable in the     If specified, the value given by
                    raw input file                   the ``units`` attribute in the
                                                     raw input file; otherwise
                                                     ``unknown``
``shift_time``      Shift time points back by 1/2 of ``True``
                    the corresponding output time
                    interval
``ugrid``           Automatic UGRIDization of        ``True``
                    the input data
``var_type``        Variable type of the             No default (needs to be specified
                    variable in the raw input        as extra facets or in recipe if
                    file                             default DRS is used)
``zg_file``         Absolute or relative (to         If possible, use geometric height
                    ``auxiliary_data_dir``) path to  at full levels provided by the raw
                    the the input file that contains input file
                    the geometric height at full
                    levels
``zg_var``          Variable name of the geometric    ``zg``
                    height at full levels in the raw
                    input file
``zghalf_file``     Absolute or relative (to         If possible, use geometric height
                    ``auxiliary_data_dir``) path to  at half levels provided by the raw
                    the the input file that contains input file
                    the geometric height at half
                    levels
``zghalf_var``      Variable name of the geometric   ``zghalf``
                    height at half levels in the raw
                    input file
=================== ================================ ===================================

.. hint::

   In order to read cell area files (``areacella`` and ``areacello``), one
   additional manual step is necessary:
   Copy the ICON grid file (you can find a download link in the global
   attribute ``grid_file_uri`` of your ICON data) to your ICON input directory
   and change its name in such a way that only the grid file is found when the
   cell area variables are required.
   Make sure that this file is not found when other variables are loaded.

   For example, you could use a new ``var_type``, e.g., ``horizontalgrid`` for
   this file.
   Thus, an ICON grid file located in
   ``2.6.1_atm_amip_R2B5_r1i1p1f1/2.6.1_atm_amip_R2B5_r1i1p1f1_horizontalgrid.nc``
   can be found using ``var_type: horizontalgrid`` in the recipe (assuming the
   default naming conventions listed above).
   Make sure that no other variable uses this ``var_type``.

   If you want to use the :func:`~esmvalcore.preprocessor.area_statistics`
   preprocessor on *regridded* ICON data, make sure to **not** use the cell area
   files by using the ``skip: true`` syntax in the recipe as described in
   :ref:`preprocessors_using_supplementary_variables`, e.g.,

   .. code-block:: yaml

     datasets:
       - {project: ICON, dataset: ICON, exp: amip,
          supplementary_variables: [{short_name: areacella, skip: true}]}


.. _read_ipsl-cm6:

IPSL-CM6
^^^^^^^^

Both output formats (i.e. the ``Output`` and the ``Analyse / Time series``
formats) are supported, and should be configured in recipes as e.g.:

.. code-block:: yaml

  datasets:
    - {simulation: CM61-LR-hist-03.1950, exp: piControl, out: Analyse, freq: TS_MO,
       account: p86caub,  status: PROD, dataset: IPSL-CM6, project: IPSLCM,
       root: /thredds/tgcc/store}
    - {simulation: CM61-LR-hist-03.1950, exp: historical, out: Output, freq: MO,
       account: p86caub,  status: PROD, dataset: IPSL-CM6, project: IPSLCM,
       root: /thredds/tgcc/store}

and data can be found by running ``esmvaltool config copy data-native-ipslcm.yml``
and adapting the ``rootpath`` to your system.

.. _ipslcm_extra_facets_example:

The ``Output`` format is an example of a case where variables are grouped in
multi-variable files, which name cannot be computed directly from datasets
attributes alone but requires the usage :ref:`config-extra-facets`.
The configuration file :download:`extra_facets_ipslcm.yml
</../esmvalcore/config/configurations/defaults/extra_facets_ipslcm.yml>`
contains the default extra facets.
These multi-variable files must also undergo some data selection.

.. _read_access-esm:

ACCESS-ESM
^^^^^^^^^^

ESMValTool can read native `ACCESS-ESM <https://research.csiro.au/access/about/esm1-5/>`__
model output.

.. warning::

  This is the first version of ACCESS-ESM CMORizer for ESMValCore. Currently,
  Supported variables: ``pr``, ``ps``, ``psl``, ``rlds``, ``tas``, ``ta``, ``va``,
  ``ua``, ``zg``, ``hus``, ``clt``, ``rsus``, ``rlus``.

The default naming conventions for input directories and files for ACCESS output are

* input directories: ``{institute}/{sub_dataset}/{exp}/{modeling_realm}/netCDF``
* input files: ``{sub_dataset}.{special_attr}-*.nc``

as configured in:

.. literalinclude:: ../configurations/data-native-access.yml
    :language: yaml
    :caption: Contents of ``data-native-access.yml``

To use this configuration, run ``esmvaltool config copy data-native-access.yml`` and
adapt the ``rootpath`` to your system.

Thus, example dataset entries could look like this:

.. code-block:: yaml

  dataset:
    - {project: ACCESS, mip: Amon, dataset:ACCESS_ESM1_5, sub_dataset: HI-CN-05,
      exp: history, modeling_realm: atm, special_attr: pa, start_year: 1986, end_year: 1986}


Similar to any other fix, the ACCESS-ESM fix allows the use of :ref:`extra
facets<config-extra-facets>`.
The configuration file :download:`extra_facets_access.yml
</../esmvalcore/config/configurations/defaults/extra_facets_access.yml>`
contains the defaults.
For some variables, extra facets are necessary; otherwise ESMValCore cannot
read them properly.
Supported keys for extra facets are:

==================== ========================================== ====================================
Key                  Description                                Default value if not specified
==================== ========================================== ====================================
``raw_name``         Variable name of the variable in the       CMOR variable name of the
                     raw input file                             corresponding variable
``modeling_realm``   Realm attribute includes `atm`, `ice`,     No default (needs to be
                     and `oce`                                  specified as extra facets or in
                                                                recipe if default DRS is used)
``freq_attribute``   A special attribute in the filename        No default
                     `ACCESS-ESM` raw data, related to the
                     frequency of raw data
``sub_dataset``      Part of the ACCESS-ESM raw dataset root,   No default
                     needs to be specified if you want to use
                     the cmoriser
``ocean_grid_path``  Path to load the grid data for ACCESS      No default
                     ocean variables
==================== ========================================== ====================================


.. _data-retrieval:

Data retrieval
==============

Please go to :ref:`config-data-sources` for instructions and background on how
to configure data retrieval.

Data loading
============

Data loading is done using the data load functionality of `iris`; we will not go into too much detail
about this since we can point the user to the specific functionality
`here <https://scitools-iris.readthedocs.io/en/latest/userguide/loading_iris_cubes.html>`_ but we will underline
that the initial loading is done by adhering to the CF Conventions that `iris` operates by as well (see
`CF Conventions Document <http://cfconventions.org/cf-conventions/cf-conventions.html>`_ and the search
page for CF `standard names <https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html>`_).

Data concatenation from multiple sources
========================================

Oftentimes data retrieving results in assembling a continuous data stream from
multiple files or even, multiple experiments. The internal mechanism through which
the assembly is done is via cube concatenation. One peculiarity of iris concatenation
(see `iris cube concatenation <https://scitools-iris.readthedocs.io/en/latest/userguide/merge_and_concat.html>`_)
is that it doesn't allow for concatenating time-overlapping cubes; this case is rather
frequent with data from models overlapping in time, and is accounted for by a function that performs a
flexible concatenation between two cubes, depending on the particular setup:

* cubes overlap in time: resulting cube is made up of the overlapping data plus left and
  right hand sides on each side of the overlapping data; note that in the case of the cubes
  coming from different experiments the resulting concatenated cube will have composite data
  made up from multiple experiments: assume [cube1: exp1, cube2: exp2] and cube1 starts before cube2,
  and cube2 finishes after cube1, then the concatenated cube will be made up of cube2: exp2 plus the
  section of cube1: exp1 that contains data not provided in cube2: exp2;
* cubes don't overlap in time: data from the two cubes is bolted together;

Note that two cube concatenation is the base operation of an iterative process of reducing multiple cubes
from multiple data segments via cube concatenation ie if there is no time-overlapping data, the
cubes concatenation is performed in one step.

.. _extra-facets-data-finder:

Use of extra facets in the datafinder
=====================================
Extra facets are a mechanism to provide additional information for certain
kinds of data. The general approach is described in :ref:`config-extra-facets`.
Here, we describe how they can be used to locate data files within the
datafinder framework.
This is useful to build paths for directory structures and file names
that require more information than what is provided in the recipe.
A common application is the location of variables in multi-variable files as
often found in climate models' native output formats.

Another use case is files that use different names for variables in their
file name than for the netCDF4 variable name.

To apply the extra facets for this purpose, simply use the corresponding tag in
the applicable ``filename_template`` or ``dirname_template`` in
:class:`esmvalcore.local.LocalDataSource`.

For example, given the extra facets

.. code-block:: yaml

  projects:
    native6:
      extra_facets:
        ERA5:
          Amon:
            tas:
              source_var_name: t2m

a corresponding entry in the configuration file could look like:

.. literalinclude:: ../configurations/data-local-esmvaltool.yml
    :language: yaml
    :caption: Contents of ``data-local-esmvaltool.yml``
    :start-at: # Data that can be read in its native format by ESMValCore.
    :end-before: # Data that has been CMORized by ESMValTool according to the CMIP6 standard.

The same replacement mechanism can be employed everywhere where tags can be
used, particularly in ``dirname_template`` and ``filename_template`` in
:class:`esmvalcore.local.LocalDataSource`, and in ``output_file`` in
:ref:`config-developer.yml <config-developer>`.
