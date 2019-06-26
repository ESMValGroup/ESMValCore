:: _inputdata:

***********
Data finder
***********

Overview
========
Data discovery and retrieval is the first step in any evaluation process; ESMValTool
uses a `semi-automated` data finding mechanism performed by the ``_data_finder.py`` module
with inputs from both the user configuration file and the recipe file. The reason why the data
finder module is `semi`-automated is that the user will have to provide the tool with a set
of parameters related to the data needed; the reason why it is semi-`automated` is that once
these parameters have been provided, the tool will automatically find the right data. We will
detail below the data finding and retrieval process and the inputs the user needs to specify,
giving examples on how to use the data finding routine under different scenarios.

CMIP data: Data Reference Syntax (DRS) and the ESGF
===================================================
CMIP data is widely available via the Earth System Grid Federation (`ESGF <https://esgf.llnl.gov/>`_)
and is accessible to users either via dowload from the ESGF portal or through the ESGF data nodes hosted
by large computing facilities (like CEDA-Jasmin, DKRZ etc). This data adheres to, among other standards,
the DRS and Controlled Vocabulary standard for naming files and structured paths; the `DRS <https://www.ecmwf.int/sites/default/files/elibrary/2014/13713-data-reference-syntax-governing-standards-within-climate-research-data-archived-esgf.pdf>`_
ensures that files and paths to them are named according to a standardized convention. An example of this
convention, and also used by ESMValTool for file discovery and data retrieval can be seen here:

* CMIP6 file: ``[variable_short_name]_[mip]_[dataset_name]_[experiment]_[ensemble]_[grid]_[start-date]-[end-date].nc``;
* CMIP5 file: ``[variable_short_name]_[mip]_[dataset_name]_[experiment]_[ensemble]_[start-date]-[end-date].nc``;

and similar standards exist for the standard paths (input directories); for the ESGF data nodes,
these paths differ slightly, an example is given below:

* CMIP6 path for BADC: ``ROOT-BADC/[institute]/[dataset_name]/[experiment]/[ensemble]/[mip]/
  [variable_short_name]/[grid]``;
* CMIP6 pth for ETHZ: ``ROOT-ETHZ/[experiment]/[mip]/[variable_short_name]/[dataset_name]/[ensemble]/[grid]``;

From the ESMValTool user perspective the number of data input parameters is optimized to allow for ease of use.
We detail this procedure in the next section.

ESMValTool data retrieval
=========================
Data retrieval in ESMValTool has two main aspects from the user's point of view:

* data can be found by the tool, subject to availability on disk;
* it is the user's responsibility to set the corect data retrieval parameters;

The first point is self-explanatory: if the user runs the tool on a machine that has access to a data
repository or multiple data repositories, then ESMValTool will look for and find the avaialble data requested
by the user.

The second point underlines the fact that the user has full control over what type and the amount of data they
need for their analyses. Setting the data retrieval parameters is explained below:

Setting the correct paths
-------------------------
The first step towards providing ESMValTool the correct set of parameters for data retrieval is setting
the root paths to the data. This is done in the user configuration file ``config-user.yml``.
The two sections where the user will set the paths are ``rootpath`` and ``drs``. ``rootpath`` contains pointers
to ``CMIP``, ``OBS``, ``default`` and ``RAWOBS`` root paths:

* ``CMIP`` e.g. ``CMIP5`` or ``CMIP6``: this is the `root` path(s) to where the CMIP files are stored;
  it can be a single path or a list of paths; it can point to an ESGF node or it can point to a user
  private repository;

  Example for a CMIP5 root path pointing to the ESGF node on CEDA-Jasmin (formerly known as BADC):

  .. code-block:: yaml

    CMIP5: /badc/cmip5/data/cmip5/output1

  Example for a CMIP6 root path pointing to the ESGF node on CEDA-Jasmin (formerly known as BADC):

  .. code-block:: yaml

    CMIP6: /badc/cmip6/data/CMIP6/CMIP

  Example for a mix of CMIP6 root path pointing to the ESGF node on CEDA-Jasmin (formerly known as BADC)
  and a user-specific data repository for extra data:

  .. code-block:: yaml

    CMIP6: [/badc/cmip6/data/CMIP6/CMIP, /home/users/joepesci/cmip_data]

* ``OBS``: this is the `root` path(s) to where the observational datasets are stored; again, this could
  be a single path or a list of paths, just like for CMIP data.

  Example for the OBS path for a large cache of observation datasets on CEDA-Jasmin:

  .. code-block:: yaml

    OBS: /group_workspaces/jasmin4/esmeval/obsdata-v2



















