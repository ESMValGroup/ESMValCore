.. _findingdata:

**********
Input data
**********

Overview
========
Data discovery and retrieval is the first step in any evaluation process;
ESMValTool uses a `semi-automated` data finding mechanism with inputs from both
the user configuration file and the recipe file: this means that the user will
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
ESMValTool for file discovery and data retrieval, include:

* CMIP6 file: ``[variable_short_name]_[mip]_[dataset_name]_[experiment]_[ensemble]_[grid]_[start-date]-[end-date].nc``
* CMIP5 file: ``[variable_short_name]_[mip]_[dataset_name]_[experiment]_[ensemble]_[start-date]-[end-date].nc``
* OBS file: ``[project]_[dataset_name]_[type]_[version]_[mip]_[short_name]_[start-date]-[end-date].nc``

Similar standards exist for the standard paths (input directories); for the
ESGF data nodes, these paths differ slightly, for example:

* CMIP6 path for BADC: ``ROOT-BADC/[institute]/[dataset_name]/[experiment]/[ensemble]/[mip]/
  [variable_short_name]/[grid]``;
* CMIP6 path for ETHZ: ``ROOT-ETHZ/[experiment]/[mip]/[variable_short_name]/[dataset_name]/[ensemble]/[grid]``

From the ESMValTool user perspective the number of data input parameters is
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

Since observational data are organized in Tiers depending on their level of
public availability, the ``default`` directory must be structured accordingly
with sub-directories ``TierX`` (``Tier1``, ``Tier2`` or ``Tier3``), even when
``drs: default``.


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
you have configured for the ``native6`` project in your :ref:`user
configuration file`, in a subdirectory called
``Tier{tier}/{dataset}/{version}/{frequency}/{short_name}``.
Replace the items in curly braces by the values used in the variable/dataset
definition in the :ref:`recipe <recipe_overview>`.
Below is a list of native reanalysis/observational datasets currently
supported.

.. _read_native_era5:

ERA5
^^^^

- Supported variables: ``clt``, ``evspsbl``, ``evspsblpot``, ``mrro``, ``pr``, ``prsn``, ``ps``, ``psl``, ``ptype``, ``rls``, ``rlds``, ``rsds``, ``rsdt``, ``rss``, ``uas``, ``vas``, ``tas``, ``tasmax``, ``tasmin``, ``tdps``, ``ts``, ``tsn`` (``E1hr``/``Amon``), ``orog`` (``fx``)
- Tier: 3

.. _read_native_mswep:

MSWEP
^^^^^

- Supported variables: ``pr``
- Supported frequencies: ``mon``, ``day``, ``3hr``.
- Tier: 3

For example for monthly data, place the files in the ``/Tier3/MSWEP/latestversion/mon/pr`` subdirectory of your ``native6`` project location.

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

ESMValTool is able to read native `CESM <https://www.cesm.ucar.edu/>`__ model
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
   * ``[case]/[gcomp]/hist`` (short-term archiving)
   * ``[case]/[gcomp]/proc/[tdir]/[tperiod]`` (post-processed data)
* input files: ``[case].[scomp].[type].[string]*nc``

as configured in the :ref:`config-developer file <config-developer>` (using the
default DRS ``drs: default`` in the :ref:`user configuration file`).
More information about CESM naming conventions are given `here
<https://www.cesm.ucar.edu/models/cesm2/naming_conventions.html>`__.

.. note::

   The ``[string]`` entry in the input file names above does not only
   correspond to the (optional) ``$string`` entry for `CESM model output files
   <https://www.cesm.ucar.edu/models/cesm2/naming_conventions.html#modelOutputFilenames>`__,
   but can also be used to read `post-processed files
   <https://www.cesm.ucar.edu/models/cesm2/naming_conventions.html#ppDataFilenames>`__.
   In the latter case, ``[string]`` corresponds to the combination
   ``$SSTRING.$TSTRING``.

Thus, example dataset entries could look like this:

.. code-block:: yaml

  datasets:
    - {project: CESM, dataset: CESM2, case: f.e21.FHIST_BGC.f09_f09_mg17.CMIP6-AMIP.001, type: h0, mip: Amon, short_name: tas, start_year: 2000, end_year: 2014}
    - {project: CESM, dataset: CESM2, case: f.e21.F1850_BGC.f09_f09_mg17.CFMIP-hadsst-piForcing.001, type: h0, gcomp: atm, scomp: cam, mip: Amon, short_name: tas, start_year: 2000, end_year: 2014}

Variable-specific defaults for the facet ``gcomp`` and ``scomp`` are given in
the extra facets (see next paragraph) for some variables, but this can be
overwritten in the recipe.

Similar to any other fix, the CESM fix allows the use of :ref:`extra
facets<extra_facets>`.
By default, the file :download:`cesm-mappings.yml
</../esmvalcore/_config/extra_facets/cesm-mappings.yml>` is used for that
purpose.
Currently, this file only contains default facets for a single variable
(`tas`); for other variables, these entries need to be defined in the recipe.
Supported keys for extra facets are:

==================== ====================================== =================================
Key                  Description                            Default value if not specified
==================== ====================================== =================================
``gcomp``            Generic component-model name           No default (needs to be specified
                                                            in extra facets or recipe if
                                                            default DRS is used)
``raw_name``         Variable name of the variable in the   CMOR variable name of the
                     raw input file                         corresponding variable
``scomp``            Specific component-model name          No default (needs to be specified
                                                            in extra facets or recipe if
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

ESMValTool is able to read native `EMAC
<https://www.dlr.de/pa/en/desktopdefault.aspx/tabid-8859/15306_read-37415/>`_
model output.

The default naming conventions for input directories and files for EMAC are

* input directories: ``[exp]/[channel]``
* input files: ``[exp]*[channel][postproc_flag].nc``

as configured in the :ref:`config-developer file <config-developer>` (using the
default DRS ``drs: default`` in the :ref:`user configuration file`).

Thus, example dataset entries could look like this:

.. code-block:: yaml

  datasets:
    - {project: EMAC, dataset: EMAC, exp: historical, mip: Amon, short_name: tas, start_year: 2000, end_year: 2014}
    - {project: EMAC, dataset: EMAC, exp: historical, mip: Omon, short_name: tos, postproc_flag: "-p-mm", start_year: 2000, end_year: 2014}
    - {project: EMAC, dataset: EMAC, exp: historical, mip: Amon, short_name: ta, raw_name: tm1_p39_cav, start_year: 2000, end_year: 2014}

Please note the duplication of the name ``EMAC`` in ``project`` and
``dataset``, which is necessary to comply with ESMValTool's data finding and
CMORizing functionalities.
A variable-specific default for the facet ``channel`` is given in the extra
facets (see next paragraph) for many variables, but this can be overwritten in
the recipe.

Similar to any other fix, the EMAC fix allows the use of :ref:`extra
facets<extra_facets>`.
By default, the file :download:`emac-mappings.yml
</../esmvalcore/_config/extra_facets/emac-mappings.yml>` is used for that
purpose.
For some variables, extra facets are necessary; otherwise ESMValTool cannot
read them properly.
Supported keys for extra facets are:

==================== ====================================== =================================
Key                  Description                            Default value if not specified
==================== ====================================== =================================
``channel``          Channel in which the desired variable  No default (needs to be specified
                     is stored                              in extra facets or recipe if
                                                            default DRS is used)
``postproc_flag``    Postprocessing flag of the data        ``''`` (empty string)
``raw_name``         Variable name of the variable in the   CMOR variable name of the
                     raw input file                         corresponding variable
==================== ====================================== =================================

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
   ``tm1_p19_ave``) are given in the default extra facets file.
   If other pressure levels are desired, e.g., ``tm1_p39_cav``, this has to be
   explicitly specified in the recipe using ``raw_name: tm1_p39_cav`` or
   ``raw_name: [tm1_p19_cav, tm1_p39_cav]``.

.. _read_icon:

ICON
^^^^

ESMValTool is able to read native `ICON
<https://code.mpimet.mpg.de/projects/iconpublic>`_ model output.

The default naming conventions for input directories and files for ICON are

* input directories: ``[exp]`` or ``{exp}/outdata``
* input files: ``[exp]_[var_type]*.nc``

as configured in the :ref:`config-developer file <config-developer>` (using the
default DRS ``drs: default`` in the :ref:`user configuration file`).

Thus, example dataset entries could look like this:

.. code-block:: yaml

  datasets:
    - {project: ICON, dataset: ICON, exp: icon-2.6.1_atm_amip_R2B5_r1i1p1f1,
       mip: Amon, short_name: tas, start_year: 2000, end_year: 2014}
    - {project: ICON, dataset: ICON, exp: historical, mip: Amon,
       short_name: ta, var_type: atm_dyn_3d_ml, start_year: 2000,
       end_year: 2014}

Please note the duplication of the name ``ICON`` in ``project`` and
``dataset``, which is necessary to comply with ESMValTool's data finding and
CMORizing functionalities.
A variable-specific default for the facet ``var_type`` is given in the extra
facets (see next paragraph) for many variables, but this can be overwritten in
the recipe.

Similar to any other fix, the ICON fix allows the use of :ref:`extra
facets<extra_facets>`.
By default, the file :download:`icon-mappings.yml
</../esmvalcore/_config/extra_facets/icon-mappings.yml>` is used for that
purpose.
For some variables, extra facets are necessary; otherwise ESMValTool cannot
read them properly.
Supported keys for extra facets are:

============= ============================= =================================
Key           Description                   Default value if not specified
============= ============================= =================================
``latitude``  Standard name of the latitude ``latitude``
              coordinate in the raw input
              file
``longitude`` Standard name of the          ``longitude``
              longitude coordinate in the
              raw input file
``raw_name``  Variable name of the          CMOR variable name of the
              variable in the raw input     corresponding variable
              file
``var_type``  Variable type of the          No default (needs to be specified
              variable in the raw input     in extra facets or recipe if
              file                          default DRS is used)
============= ============================= =================================

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

.. _ipslcm_extra_facets_example:

The ``Output`` format is an example of a case where variables are grouped in
multi-variable files, which name cannot be computed directly from datasets
attributes alone but requires to use an extra_facets file, which principles are
explained in :ref:`extra_facets`, and which content is :download:`available here
</../esmvalcore/_config/extra_facets/ipslcm-mappings.yml>`. These multi-variable
files must also undergo some data selection.


.. _data-retrieval:

Data retrieval
==============
Data retrieval in ESMValTool has two main aspects from the user's point of
view:

* data can be found by the tool, subject to availability on disk or `ESGF <https://esgf.llnl.gov/>`_;
* it is the user's responsibility to set the correct data retrieval parameters;

The first point is self-explanatory: if the user runs the tool on a machine
that has access to a data repository or multiple data repositories, then
ESMValTool will look for and find the available data requested by the user.
If the files are not found locally, the tool can search the ESGF_ and download
the missing files, provided that they are available.

The second point underlines the fact that the user has full control over what
type and the amount of data is needed for the analyses. Setting the data
retrieval parameters is explained below.

Enabling automatic downloads from the ESGF
------------------------------------------
To enable automatic downloads from ESGF, set ``offline: false`` in
the :ref:`user configuration file` or provide the command line argument
``--offline=False`` when running the recipe.
The files will be stored in the ``download_dir`` set in
the :ref:`user configuration file`.

Setting the correct root paths
------------------------------
The first step towards providing ESMValTool the correct set of parameters for
data retrieval is setting the root paths to the data. This is done in the user
configuration file ``config-user.yml``. The two sections where the user will
set the paths are ``rootpath`` and ``drs``. ``rootpath`` contains pointers to
``CMIP``, ``OBS``, ``default`` and ``RAWOBS`` root paths; ``drs`` sets the type
of directory structure the root paths are structured by. It is important to
first discuss the ``drs`` parameter: as we've seen in the previous section, the
DRS as a standard is used for both file naming conventions and for directory
structures.

Synda
-----

If the `synda install <https://prodiguer.github.io/synda/sdt/user_guide.html#synda-install>`_ command is used to download data,
it maintains the directory structure as on ESGF. To find data downloaded by
synda, use the ``SYNDA`` ``drs`` parameter.

.. code-block:: yaml

 drs:
   CMIP6: SYNDA
   CMIP5: SYNDA

.. _config-user-drs:

Explaining ``config-user/drs: CMIP5:`` or ``config-user/drs: CMIP6:``
---------------------------------------------------------------------
Whereas ESMValTool will **always** use the CMOR standard for file naming (please
refer above), by setting the ``drs`` parameter the user tells the tool what
type of root paths they need the data from, e.g.:

  .. code-block:: yaml

   drs:
     CMIP6: BADC

will tell the tool that the user needs data from a repository structured
according to the BADC DRS structure, i.e.:

``ROOT/[institute]/[dataset_name]/[experiment]/[ensemble]/[mip]/[variable_short_name]/[grid]``;

setting the ``ROOT`` parameter is explained below. This is a
strictly-structured repository tree and if there are any sort of irregularities
(e.g. there is no ``[mip]`` directory) the data will not be found! ``BADC`` can
be replaced with ``DKRZ`` or ``ETHZ`` depending on the existing ``ROOT``
directory structure.
The snippet

  .. code-block:: yaml

   drs:
     CMIP6: default

is another way to retrieve data from a ``ROOT`` directory that has no DRS-like
structure; ``default`` indicates that the data lies in a directory that
contains all the files without any structure.

.. note::
   When using ``CMIP6: default`` or ``CMIP5: default`` it is important to
   remember that all the needed files must be in the same top-level directory
   set by ``default`` (see below how to set ``default``).

.. _config-user-rootpath:

Explaining ``config-user/rootpath:``
------------------------------------

``rootpath`` identifies the root directory for different data types (``ROOT`` as we used it above):

* ``CMIP`` e.g. ``CMIP5`` or ``CMIP6``: this is the `root` path(s) to where the
  CMIP files are stored; it can be a single path or a list of paths; it can
  point to an ESGF node or it can point to a user private repository. Example
  for a CMIP5 root path pointing to the ESGF node on CEDA-Jasmin (formerly
  known as BADC):

  .. code-block:: yaml

    CMIP5: /badc/cmip5/data/cmip5/output1

  Example for a CMIP6 root path pointing to the ESGF node on CEDA-Jasmin:

  .. code-block:: yaml

    CMIP6: /badc/cmip6/data/CMIP6/CMIP

  Example for a mix of CMIP6 root path pointing to the ESGF node on CEDA-Jasmin
  and a user-specific data repository for extra data:

  .. code-block:: yaml

    CMIP6: [/badc/cmip6/data/CMIP6/CMIP, /home/users/johndoe/cmip_data]

* ``OBS``: this is the `root` path(s) to where the observational datasets are
  stored; again, this could be a single path or a list of paths, just like for
  CMIP data. Example for the OBS path for a large cache of observation datasets
  on CEDA-Jasmin:

  .. code-block:: yaml

    OBS: /gws/nopw/j04/esmeval/obsdata-v2

* ``default``: this is the `root` path(s) where the tool will look for data
  from projects that do not have their own rootpath set.

* ``RAWOBS``: this is the `root` path(s) to where the raw observational data
  files are stored; this is used by ``esmvaltool data format``.

Dataset definitions in ``recipe``
---------------------------------
Once the correct paths have been established, ESMValTool collects the
information on the specific datasets that are needed for the analysis. This
information, together with the CMOR convention for naming files (see CMOR-DRS_)
will allow the tool to search and find the right files. The specific
datasets are listed in any recipe, under either the ``datasets`` and/or
``additional_datasets`` sections, e.g.

.. code-block:: yaml

  datasets:
    - {dataset: HadGEM2-CC, project: CMIP5, exp: historical, ensemble: r1i1p1, start_year: 2001, end_year: 2004}
    - {dataset: UKESM1-0-LL, project: CMIP6, exp: historical, ensemble: r1i1p1f2, grid: gn, start_year: 2004, end_year: 2014}

``_data_finder`` will use this information to find data for **all** the variables specified in ``diagnostics/variables``.

Recap and example
=================
Let us look at a practical example for a recap of the information above:
suppose you are using a ``config-user.yml`` that has the following entries for
data finding:

.. code-block:: yaml

  rootpath:  # running on CEDA-Jasmin
    CMIP6: /badc/cmip6/data/CMIP6/CMIP
  drs:
    CMIP6: BADC  # since you are on CEDA-Jasmin

and the dataset you need is specified in your ``recipe.yml`` as:

.. code-block:: yaml

  - {dataset: UKESM1-0-LL, project: CMIP6, mip: Amon, exp: historical, grid: gn, ensemble: r1i1p1f2, start_year: 2004, end_year: 2014}

for a variable, e.g.:

.. code-block:: yaml

  diagnostics:
    some_diagnostic:
      description: some_description
      variables:
        ta:
          preprocessor: some_preprocessor

The tool will then use the root path ``/badc/cmip6/data/CMIP6/CMIP`` and the
dataset information and will assemble the full DRS path using information from
CMOR-DRS_ and establish the path to the files as:

.. code-block:: bash

  /badc/cmip6/data/CMIP6/CMIP/MOHC/UKESM1-0-LL/historical/r1i1p1f2/Amon

then look for variable ``ta`` and specifically the latest version of the data
file:

.. code-block:: bash

  /badc/cmip6/data/CMIP6/CMIP/MOHC/UKESM1-0-LL/historical/r1i1p1f2/Amon/ta/gn/latest/

and finally, using the file naming definition from CMOR-DRS_ find the file:

.. code-block:: bash

  /badc/cmip6/data/CMIP6/CMIP/MOHC/UKESM1-0-LL/historical/r1i1p1f2/Amon/ta/gn/latest/ta_Amon_UKESM1-0-LL_historical_r1i1p1f2_gn_195001-201412.nc

.. _observations:


Data loading
============

Data loading is done using the data load functionality of `iris`; we will not go into too much detail
about this since we can point the user to the specific functionality
`here <https://scitools-iris.readthedocs.io/en/latest/userguide/loading_iris_cubes.html>`_ but we will underline
that the initial loading is done by adhering to the CF Conventions that `iris` operates by as well (see
`CF Conventions Document <http://cfconventions.org/cf-conventions/cf-conventions.html>`_ and the search
page for CF `standard names <http://cfconventions.org/standard-names.html>`_).

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
Extra facets are a mechanism to provide additional information for certain kinds
of data. The general approach is described in :ref:`extra_facets`. Here, we
describe how they can be used to locate data files within the datafinder
framework.
This is useful to build paths for directory structures and file names
that require more information than what is provided in the recipe.
A common application is the location of variables in multi-variable files as
often found in climate models' native output formats.

Another use case is files that use different names for variables in their
file name than for the netCDF4 variable name.

To apply the extra facets for this purpose, simply use the corresponding tag in
the applicable DRS inside the `config-developer.yml` file. For example, given
the extra facets in :ref:`extra-facets-example-1`, one might write the
following.

.. _extra-facets-example-2:

.. code-block:: yaml
   :caption: Example drs use in `config-developer.yml`

   native6:
     input_file:
       default: '{name_in_filename}*.nc'

The same replacement mechanism can be employed everywhere where tags can be
used, particularly in `input_dir` and `input_file`.
