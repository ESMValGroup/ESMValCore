.. _findingdata:

************
Finding data
************

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

.. _CMOR-DRS:

CMIP data - CMOR Data Reference Syntax (DRS) and the ESGF
=========================================================
CMIP data is widely available via the Earth System Grid Federation
(`ESGF <https://esgf.llnl.gov/>`_) and is accessible to users either
via dowload from the ESGF portal or through the ESGF data nodes hosted
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

.. _data-retrieval:

Data retrieval
==============
Data retrieval in ESMValTool has two main aspects from the user's point of
view: 

* data can be found by the tool, subject to availability on disk;
* it is the user's responsibility to set the correct data retrieval parameters;

The first point is self-explanatory: if the user runs the tool on a machine
that has access to a data repository or multiple data repositories, then
ESMValTool will look for and find the avaialble data requested by the user.

The second point underlines the fact that the user has full control over what
type and the amount of data is needed for the analyses. Setting the data
retrieval parameters is explained below.

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

.. _config-user-drs:

Explaining ``config-user/drs: CMIP5:`` or ``config-user/drs: CMIP6:``
---------------------------------------------------------------------
Whreas ESMValTool will **always** use the CMOR standard for file naming (please
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

    OBS: /group_workspaces/jasmin4/esmeval/obsdata-v2

* ``default``: this is the `root` path(s) to where files are stored without any
  DRS-like directory structure; in a nutshell, this is a single directory that
  should contain all the files needed by the run, without any sub-directory
  structure. 

* ``RAWOBS``: this is the `root` path(s) to where the raw observational data
  files are stored; this is used by ``cmorize_obs``.

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
    - {dataset: HadGEM2-CC,  project: CMIP5, exp: historical, ensemble: r1i1p1, start_year: 2001, end_year: 2004}
    - {dataset: UKESM1-0-LL, project: CMIP6, exp: historical, ensemble: r1i1p1f2, grid: gn, start_year: 2004,  end_year: 2014}

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

  - {dataset: UKESM1-0-LL, project: CMIP6, mip: Amon, exp: historical, grid: gn, ensemble: r1i1p1f2, start_year: 2004,  end_year: 2014}

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

.. code-block::

  /badc/cmip6/data/CMIP6/CMIP/MOHC/UKESM1-0-LL/historical/r1i1p1f2/Amon

then look for variable ``ta`` and specifically the latest version of the data
file: 

.. code-block::

  /badc/cmip6/data/CMIP6/CMIP/MOHC/UKESM1-0-LL/historical/r1i1p1f2/Amon/ta/gn/latest/

and finally, using the file naming definition from CMOR-DRS_ find the file:

.. code-block::

  /badc/cmip6/data/CMIP6/CMIP/MOHC/UKESM1-0-LL/historical/r1i1p1f2/Amon/ta/gn/latest/ta_Amon_UKESM1-0-LL_historical_r1i1p1f2_gn_195001-201412.nc

.. _observations:

Observational data
==================
Observational data is retrieved in the same manner as CMIP data, for example
using the ``OBS`` root path set to:

  .. code-block:: yaml

    OBS: /group_workspaces/jasmin4/esmeval/obsdata-v2

and the dataset:

  .. code-block:: yaml

    - {dataset: ERA-Interim,  project: OBS,  type: reanaly,  version: 1,  start_year: 2014,  end_year: 2015,  tier: 3}

in ``recipe.yml`` in ``datasets`` or ``additional_datasets``, the rules set in
CMOR-DRS_ are used again and the file will be automatically found:

.. code-block::

  /group_workspaces/jasmin4/esmeval/obsdata-v2/Tier3/ERA-Interim/OBS_ERA-Interim_reanaly_1_Amon_ta_201401-201412.nc

Since observational data are organized in Tiers depending on their level of
public availability, the ``default`` directory must be structured accordingly
with sub-directories ``TierX`` (``Tier1``, ``Tier2`` or ``Tier3``), even when
``drs: default``.
