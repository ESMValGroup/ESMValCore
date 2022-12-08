.. _config:

*******************
Configuration files
*******************

Overview
========

There are several configuration files in ESMValCore:

* ``config-user.yml``: sets a number of user-specific options like desired
  graphical output format, root paths to data, etc.;
* ``config-developer.yml``: sets a number of standardized file-naming and paths
  to data formatting;

and one configuration file which is distributed with ESMValTool:

* ``config-references.yml``: stores information on diagnostic and recipe authors and
  scientific journals references;

.. _user configuration file:

User configuration file
=======================


The ``config-user.yml`` configuration file contains all the global level
information needed by ESMValTool. It can be reused as many times the user needs
to before changing any of the options stored in it. This file is essentially
the gateway between the user and the machine-specific instructions to
``esmvaltool``. By default, esmvaltool looks for it in the home directory,
inside the ``.esmvaltool`` folder.

Users can get a copy of this file with default values by running

.. code-block:: bash

  esmvaltool config get-config-user --path=${TARGET_FOLDER}

If the option ``--path`` is omitted, the file will be created in
``${HOME}/.esmvaltool``

The following shows the default settings from the ``config-user.yml`` file
with explanations in a commented line above each option. If only certain values
are allowed for an option, these are listed after ``---``. The option in square
brackets is the default value, i.e., the one that is used if this option is
omitted in the file.

.. code-block:: yaml

  # Destination directory where all output will be written
  # Includes log files and performance stats.
  output_dir: ~/esmvaltool_output

  # Directory for storing downloaded climate data
  download_dir: ~/climate_data

  # Disable automatic downloads --- [true]/false
  # Disable the automatic download of missing CMIP3, CMIP5, CMIP6, CORDEX,
  # and obs4MIPs data from ESGF by default. This is useful if you are working
  # on a computer without an internet connection.
  offline: true

  # Auxiliary data directory
  # Used by some recipes to look for additional datasets.
  auxiliary_data_dir: ~/auxiliary_data

  # Rootpaths to the data from different projects
  # This default setting will work if files have been downloaded by the
  # ESMValTool via ``offline=False``. Lists are also possible. For
  # site-specific entries, see the default ``config-user.yml`` file that can be
  # installed with the command ``esmvaltool config get_config_user``. For each
  # project, this can be either a single path or a list of paths. Comment out
  # these when using a site-specific path.
  rootpath:
    default: ~/climate_data

  # Directory structure for input data --- [default]/ESGF/BADC/DKRZ/ETHZ/etc.
  # This default setting will work if files have been downloaded by the
  # ESMValTool via ``offline=False``. See ``config-developer.yml`` for
  # definitions. Comment out/replace as per needed.
  drs:
    CMIP3: ESGF
    CMIP5: ESGF
    CMIP6: ESGF
    CORDEX: ESGF
    obs4MIPs: ESGF

  # Run at most this many tasks in parallel --- [null]/1/2/3/4/...
  # Set to ``null`` to use the number of available CPUs. If you run out of
  # memory, try setting max_parallel_tasks to ``1`` and check the amount of
  # memory you need for that by inspecting the file ``run/resource_usage.txt`` in
  # the output directory. Using the number there you can increase the number of
  # parallel tasks again to a reasonable number for the amount of memory
  # available in your system.
  max_parallel_tasks: null

  # Log level of the console --- debug/[info]/warning/error
  # For much more information printed to screen set log_level to ``debug``.
  log_level: info

  # Exit on warning --- true/[false]
  # Only used in NCL diagnostic scripts.
  exit_on_warning: false

  # Plot file format --- [png]/pdf/ps/eps/epsi
  output_file_type: png

  # Remove the ``preproc`` directory if the run was successful --- [true]/false
  # By default this option is set to ``true``, so all preprocessor output files
  # will be removed after a successful run. Set to ``false`` if you need those files.
  remove_preproc_dir: true

  # Use netCDF compression --- true/[false]
  compress_netcdf: false

  # Save intermediary cubes in the preprocessor --- true/[false]
  # Setting this to ``true`` will save the output cube from each preprocessing
  # step. These files are numbered according to the preprocessing order.
  save_intermediary_cubes: false

  # Use a profiling tool for the diagnostic run --- [false]/true
  # A profiler tells you which functions in your code take most time to run.
  # For this purpose we use ``vprof``, see below for notes. Only available for
  # Python diagnostics.
  profile_diagnostic: false

  # Path to custom ``config-developer.yml`` file
  # This can be used to customise project configurations. See
  # ``config-developer.yml`` for an example. Set to ``null`` to use the default.
  config_developer_file: null

The ``offline`` setting can be used to disable or enable automatic downloads from ESGF.
If ``offline`` is set to ``false``, the tool will automatically download
any CMIP3, CMIP5, CMIP6, CORDEX, and obs4MIPs data that is required to run a recipe
but not available locally and store it in ``download_dir`` using the ``ESGF``
directory structure defined in the :ref:`config-developer`.

The ``auxiliary_data_dir`` setting is the path to place any required
additional auxiliary data files. This is necessary because certain
Python toolkits, such as cartopy, will attempt to download data files at run
time, typically geographic data files such as coastlines or land surface maps.
This can fail if the machine does not have access to the wider internet. This
location allows the user to specify where to find such files if they can not be
downloaded at runtime. The example user configuration file already contains two valid
locations for ``auxiliary_data_dir`` directories on CEDA-JASMIN and DKRZ, and a number
of such maps and shapefiles (used by current diagnostics) are already there. You will
need ``esmeval`` group workspace membership to access the JASMIN one (see
`instructions <https://help.jasmin.ac.uk/article/199-introduction-to-group-workspaces>`_
how to gain access to the group workspace.

.. warning::

   This setting is not for model or observational datasets, rather it is for
   extra data files such as shapefiles or other data sources needed by the diagnostics.

The ``profile_diagnostic`` setting triggers profiling of Python diagnostics,
this will tell you which functions in the diagnostic took most time to run.
For this purpose we use `vprof <https://github.com/nvdv/vprof>`_.
For each diagnostic script in the recipe, the profiler writes a ``.json`` file
that can be used to plot a
`flame graph <https://queue.acm.org/detail.cfm?id=2927301>`__
of the profiling information by running

.. code-block:: bash

  vprof --input-file esmvaltool_output/recipe_output/run/diagnostic/script/profile.json

Note that it is also possible to use vprof to understand other resources used
while running the diagnostic, including execution time of different code blocks
and memory usage.

A detailed explanation of the data finding-related sections of the
``config-user.yml`` (``rootpath`` and ``drs``) is presented in the
:ref:`data-retrieval` section. This section relates directly to the data
finding capabilities  of ESMValTool and are very important to be understood by
the user.

.. note::

   You can choose your ``config-user.yml`` file at run time, so you could have several of
   them available with different purposes. One for a formalised run, another for
   debugging, etc. You can even provide any config user value as a run flag
   ``--argument_name argument_value``


.. _config-esgf:

ESGF configuration
==================

The ``esmvaltool run`` command can automatically download the files required
to run a recipe from ESGF for the projects CMIP3, CMIP5, CMIP6, CORDEX, and obs4MIPs.
The downloaded files will be stored in the ``download_dir`` specified in the
:ref:`user configuration file`.
To enable automatic downloads from ESGF, set ``offline: false`` in
the :ref:`user configuration file` or provide the command line argument
``--offline=False`` when running the recipe.

.. note::

   When running a recipe that uses many or large datasets on a machine that
   does not have any data available locally, the amount of data that will be
   downloaded can be in the range of a few hundred gigabyte to a few terrabyte.
   See :ref:`esmvaltool:inputdata` for advice on getting access to machines
   with large datasets already available.

   A log message will be displayed with the total amount of data that will
   be downloaded before starting the download.
   If you see that this is more than you would like to download, stop the
   tool by pressing the ``Ctrl`` and ``C`` keys on your keyboard simultaneously
   several times, edit the recipe so it contains fewer datasets and try again.

For downloading some files (e.g. those produced by the CORDEX project),
you need to log in to be able to download the data.

See the
`ESGF user guide <https://esgf.github.io/esgf-user-support/user_guide.html>`_
for instructions on how to create an ESGF OpenID account if you do not have
one yet.
Note that the OpenID account consists of 3 components instead of the usual
two, in addition a username and password you also need the hostname of the
provider of the ID; for example
`esgf-data.dkrz.de <https://esgf-data.dkrz.de/user/add/?next=http://esgf-data.dkrz.de/projects/esgf-dkrz/>`_.
Even though the account is issued by a particular host, the same OpenID
account can be used to download data from all hosts in the ESGF.

Next, configure your system so the ``esmvaltool`` can use your credentials.
This can be done using the keyring_ package or they can be stored in a
:ref:`configuration file <config_esgf_pyclient>`.

.. _keyring:

Storing credentials in keyring
------------------------------
First install the keyring package. Note that this requires a supported
backend that may not be available on compute clusters, see the
`keyring documentation <https://pypi.org/project/keyring>`__ for more
information.

.. code-block:: bash

    pip install keyring

Next, set your username and password by running the commands:

.. code-block:: bash

    keyring set ESGF hostname
    keyring set ESGF username
    keyring set ESGF password

for example, if you created an account on the host `esgf-data.dkrz.de`_ with username
'cookiemonster' and password 'Welcome01', run the command

.. code-block:: bash

    keyring set ESGF hostname

this will display the text

.. code-block:: bash

    Password for 'hostname' in 'ESGF':

type ``esgf-data.dkrz.de`` (the characters will not be shown) and press ``Enter``.
Repeat the same procedure with ``keyring set ESGF username``, type ``cookiemonster``
and press ``Enter`` and ``keyring set ESGF password``, type ``Welcome01`` and
press ``Enter``.

To check that you entered your credentials correctly, run:

.. code-block:: bash

    keyring get ESGF hostname
    keyring get ESGF username
    keyring get ESGF password

.. _config_esgf_pyclient:

Configuration file
------------------
An optional configuration file can be created for configuring how the tool uses
`esgf-pyclient <https://esgf-pyclient.readthedocs.io>`_
to find and download data.
The name of this file is ``~/.esmvaltool/esgf-pyclient.yml``.

Logon
`````
In the ``logon`` section you can provide arguments that will be passed on to
:py:meth:`pyesgf.logon.LogonManager.logon`.
For example, you can store the hostname, username, and password or your OpenID
account in the file like this:

.. code-block:: yaml

    logon:
      hostname: "your-hostname"
      username: "your-username"
      password: "your-password"

for example

.. code-block:: yaml

    logon:
      hostname: "esgf-data.dkrz.de"
      username: "cookiemonster"
      password: "Welcome01"

if you created an account on the host `esgf-data.dkrz.de`_ with username
'cookiemonster' and password 'Welcome01'.
Alternatively, you can configure an interactive log in:

.. code-block:: yaml

    logon:
      interactive: true

Note that storing your password in plain text in the configuration
file is less secure.
On shared systems, make sure the permissions of the file are set so
only you and administrators can read it, i.e.

.. code-block:: bash

    ls -l ~/.esmvaltool/esgf-pyclient.yml

shows permissions ``-rw-------``.

Search
``````
Any arguments to :py:obj:`pyesgf.search.connection.SearchConnection` can
be provided in the section ``search_connection``, for example:

.. code-block:: yaml

    search_connection:
      expire_after: 2592000  # the number of seconds in a month

to keep cached search results for a month.

The default settings are:

.. code-block:: yaml

    urls:
      - 'https://esgf-index1.ceda.ac.uk/esg-search'
      - 'https://esgf-node.llnl.gov/esg-search'
      - 'https://esgf-data.dkrz.de/esg-search'
      - 'https://esgf-node.ipsl.upmc.fr/esg-search'
      - 'https://esg-dn1.nsc.liu.se/esg-search'
      - 'https://esgf.nci.org.au/esg-search'
      - 'https://esgf.nccs.nasa.gov/esg-search'
      - 'https://esgdata.gfdl.noaa.gov/esg-search'
    distrib: true
    timeout: 120  # seconds
    cache: '~/.esmvaltool/cache/pyesgf-search-results'
    expire_after: 86400  # cache expires after 1 day

Note that by default the tool will try the
`ESGF index nodes <https://esgf.llnl.gov/nodes.html>`_
in the order provided in the configuration file and use the first one that is
online.
Some ESGF index nodes may return search results faster than others, so you may
be able to speed up the search for files by experimenting with placing different
index nodes at the top of the list.

If you experience errors while searching, it sometimes helps to delete the
cached results.

Download statistics
-------------------
The tool will maintain statistics of how fast data can be downloaded
from what host in the file ~/.esmvaltool/cache/esgf-hosts.yml and
automatically select hosts that are faster.
There is no need to manually edit this file, though it can be useful
to delete it if you move your computer to a location that is very
different from the place where you previously downloaded data.
An entry in the file might look like this:

.. code-block:: yaml

    esgf2.dkrz.de:
      duration (s): 8
      error: false
      size (bytes): 69067460
      speed (MB/s): 7.9

The tool only uses the duration and size to determine the download speed,
the speed shown in the file is not used.
If ``error`` is set to ``true``, the most recent download request to that
host failed and the tool will automatically try this host only as a last
resort.

.. _config-developer:

Developer configuration file
============================

Most users and diagnostic developers will not need to change this file,
but it may be useful to understand its content.
It will be installed along with ESMValCore and can also be viewed on GitHub:
`esmvalcore/config-developer.yml
<https://github.com/ESMValGroup/ESMValCore/blob/main/esmvalcore/config-developer.yml>`_.
This configuration file describes the file system structure and CMOR tables for several
key projects (CMIP6, CMIP5, obs4MIPs, OBS6, OBS) on several key machines (e.g. BADC, CP4CDS, DKRZ,
ETHZ, SMHI, BSC), and for native output data for some
models (ICON, IPSL, ... see :ref:`configure_native_models`).
CMIP data is stored as part of the Earth System Grid
Federation (ESGF) and the standards for file naming and paths to files are set
out by CMOR and DRS. For a detailed description of these standards and their
adoption in ESMValCore, we refer the user to :ref:`CMOR-DRS` section where we
relate these standards to the data retrieval mechanism of the ESMValCore.

By default, esmvaltool looks for it in the home directory,
inside the '.esmvaltool' folder.

Users can get a copy of this file with default values by running

.. code-block:: bash

  esmvaltool config get-config-developer --path=${TARGET_FOLDER}

If the option ``--path`` is omitted, the file will be created in
```${HOME}/.esmvaltool``.

.. note::

  Remember to change your config-user file if you want to use a custom
  config-developer.

Example of the CMIP6 project configuration:

.. code-block:: yaml

   CMIP6:
     input_dir:
       default: '/'
       BADC: '{activity}/{institute}/{dataset}/{exp}/{ensemble}/{mip}/{short_name}/{grid}/{latestversion}'
       DKRZ: '{activity}/{institute}/{dataset}/{exp}/{ensemble}/{mip}/{short_name}/{grid}/{latestversion}'
       ETHZ: '{exp}/{mip}/{short_name}/{dataset}/{ensemble}/{grid}/'
     input_file: '{short_name}_{mip}_{dataset}_{exp}_{ensemble}_{grid}*.nc'
     output_file: '{project}_{dataset}_{mip}_{exp}_{ensemble}_{short_name}'
     cmor_type: 'CMIP6'
     cmor_strict: true

Input file paths
----------------

When looking for input files, the ``esmvaltool`` command provided by
ESMValCore replaces the placeholders ``{item}`` in
``input_dir`` and ``input_file`` with the values supplied in the recipe.
ESMValCore will try to automatically fill in the values for institute, frequency,
and modeling_realm based on the information provided in the CMOR tables
and/or extra_facets_ when reading the recipe.
If this fails for some reason, these values can be provided in the recipe too.

The data directory structure of the CMIP projects is set up differently
at each site. As an example, the CMIP6 directory path on BADC would be:

.. code-block:: yaml

   '{activity}/{institute}/{dataset}/{exp}/{ensemble}/{mip}/{short_name}/{grid}/{latestversion}'

The resulting directory path would look something like this:

.. code-block:: bash

    CMIP/MOHC/HadGEM3-GC31-LL/historical/r1i1p1f3/Omon/tos/gn/latest

Please, bear in mind that ``input_dirs`` can also be a list for those  cases in
which may be needed:

.. code-block:: yaml

  - '{exp}/{ensemble}/original/{mip}/{short_name}/{grid}/{latestversion}'
  - '{exp}/{ensemble}/computed/{mip}/{short_name}/{grid}/{latestversion}'

In that case, the resultant directories will be:

.. code-block:: bash

  historical/r1i1p1f3/original/Omon/tos/gn/latest
  historical/r1i1p1f3/computed/Omon/tos/gn/latest

For a more in-depth description of how to configure ESMValCore so it can find
your data please see :ref:`CMOR-DRS`.

Preprocessor output files
-------------------------

The filename to use for preprocessed data is configured in a similar manner
using ``output_file``. Note that the extension ``.nc`` (and if applicable,
a start and end time) will automatically be appended to the filename.

.. _cmor_table_configuration:

Project CMOR table configuration
--------------------------------

ESMValCore comes bundled with several CMOR tables, which are stored in the directory
`esmvalcore/cmor/tables <https://github.com/ESMValGroup/ESMValCore/tree/main/esmvalcore/cmor/tables>`_.
These are copies of the tables available from `PCMDI <https://github.com/PCMDI>`_.

For every ``project`` that can be used in the recipe, there are four settings
related to CMOR table settings available:

* ``cmor_type``: can be ``CMIP5`` if the CMOR table is in the same format as the
  CMIP5 table or ``CMIP6`` if the table is in the same format as the CMIP6 table.
* ``cmor_strict``: if this is set to ``false``, the CMOR table will be
  extended with variables from the :ref:`custom_cmor_tables` (by default loaded
  from the ``esmvalcore/cmor/tables/custom`` directory) and it is possible to
  use variables with a ``mip`` which is different from the MIP table in which
  they are defined.
* ``cmor_path``: path to the CMOR table.
  Relative paths are with respect to `esmvalcore/cmor/tables`_.
  Defaults to the value provided in ``cmor_type`` written in lower case.
* ``cmor_default_table_prefix``: Prefix that needs to be added to the ``mip``
  to get the name of the file containing the ``mip`` table.
  Defaults to the value provided in ``cmor_type``.

.. _custom_cmor_tables:

Custom CMOR tables
------------------

As mentioned in the previous section, the CMOR tables of projects that use
``cmor_strict: false`` will be extended with custom CMOR tables.
By default, these are loaded from `esmvalcore/cmor/tables/custom
<https://github.com/ESMValGroup/ESMValCore/tree/main/esmvalcore/cmor/tables/custom>`_.
However, by using the special project ``custom`` in the
``config-developer.yml`` file with the option ``cmor_path``, a custom location
for these custom CMOR tables can be specified:

.. code-block:: yaml

   custom:
     cmor_path: ~/my/own/custom_tables

This path can be given as relative path (relative to `esmvalcore/cmor/tables`_)
or as absolute path.
Other options given for this special table will be ignored.

Custom tables in this directory need to follow the naming convention
``CMOR_{short_name}.dat`` and need to be given in CMIP5 format.

Example for the file ``CMOR_asr.dat``:

.. code-block::

   SOURCE: CMIP5
   !============
   variable_entry:    asr
   !============
   modeling_realm:    atmos
   !----------------------------------
   ! Variable attributes:
   !----------------------------------
   standard_name:
   units:             W m-2
   cell_methods:      time: mean
   cell_measures:     area: areacella
   long_name:         Absorbed shortwave radiation
   !----------------------------------
   ! Additional variable information:
   !----------------------------------
   dimensions:        longitude latitude time
   type:              real
   positive:          down
   !----------------------------------
   !

It is also possible to use a special coordinates file ``CMOR_coordinates.dat``.
If this is not present in the custom directory, the one from the default
directory (`esmvalcore/cmor/tables/custom/CMOR_coordinates.dat
<https://github.com/ESMValGroup/ESMValCore/tree/main/esmvalcore/cmor/tables/custom/CMOR_coordinates.dat>`_)
is used.


.. _filterwarnings_config-developer:

Filter preprocessor warnings
----------------------------

It is possible to ignore specific warnings of the preprocessor for a given
``project``.
This is particularly useful for native datasets which do not follow the CMOR
standard by default and consequently produce a lot of warnings when handled by
Iris.
This can be configured in the ``config-developer.yml`` file for some steps of
the preprocessing chain.

Currently supported preprocessor steps:

* :func:`~esmvalcore.preprocessor.load`

Here is an example on how to ignore specific warnings during the preprocessor
step ``load`` for all datasets of project  ``EMAC`` (taken from the default
``config-developer.yml`` file):

.. code-block:: yaml

   ignore_warnings:
     load:
       - {message: 'Missing CF-netCDF formula term variable .*, referenced by netCDF variable .*', module: iris}
       - {message: 'Ignored formula of unrecognised type: .*', module: iris}

The keyword arguments specified in the list items are directly passed to
:func:`warnings.filterwarnings` in addition to ``action=ignore`` (may be
overwritten in ``config-developer.yml``).

.. _configure_native_models:

Configuring datasets in native format
-------------------------------------

ESMValCore can be configured for handling native model output formats and
specific reanalysis/observation datasets without preliminary reformatting.
These datasets can be either hosted under the ``native6`` project (mostly
native reanalysis/observational datasets) or under a dedicated project, e.g.,
``ICON`` (mostly native models).

Example:

.. code-block:: yaml

   native6:
     cmor_strict: false
     input_dir:
       default: 'Tier{tier}/{dataset}/{latestversion}/{frequency}/{short_name}'
     input_file:
       default: '*.nc'
     output_file: '{project}_{dataset}_{type}_{version}_{mip}_{short_name}'
     cmor_type: 'CMIP6'
     cmor_default_table_prefix: 'CMIP6_'

   ICON:
     cmor_strict: false
     input_dir:
       default:
         - '{exp}'
         - '{exp}/outdata'
     input_file:
       default: '{exp}_{var_type}*.nc'
     output_file: '{project}_{dataset}_{exp}_{var_type}_{mip}_{short_name}'
     cmor_type: 'CMIP6'
     cmor_default_table_prefix: 'CMIP6_'

A detailed description on how to add support for further native datasets is
given :ref:`here <add_new_fix_native_datasets>`.

.. hint::

   When using native datasets, it might be helpful to specify a custom location
   for the :ref:`custom_cmor_tables`.
   This allows reading arbitrary variables from native datasets.
   Note that this requires the option ``cmor_strict: false`` in the
   :ref:`project configuration <configure_native_models>` used for the native
   model output.


.. _config-ref:

References configuration file
=============================

The `esmvaltool/config-references.yml <https://github.com/ESMValGroup/ESMValTool/blob/main/esmvaltool/config-references.yml>`__ file contains the list of ESMValTool diagnostic and recipe authors,
references and projects. Each author, project and reference referred to in the
documentation section of a recipe needs to be in this file in the relevant
section.

For instance, the recipe ``recipe_ocean_example.yml`` file contains the
following documentation section:

.. code-block:: yaml

  documentation:
    authors:
      - demo_le

    maintainer:
      - demo_le

    references:
      - demora2018gmd

    projects:
      - ukesm


These four items here are named people, references and projects listed in the
``config-references.yml`` file.

.. _extra_facets:

Extra Facets
============

It can be useful to automatically add extra key-value pairs to variables
or datasets in the recipe.
These key-value pairs can be used for :ref:`finding data <findingdata>`
or for providing extra information to the functions that
:ref:`fix data <extra-facets-fixes>` before passing it on to the preprocessor.

To support this, we provide the extra facets facilities. Facets are the
key-value pairs described in :ref:`Datasets`. Extra facets allows for the
addition of more details per project, dataset, mip table, and variable name.

More precisely, one can provide this information in an extra yaml file, named
`{project}-something.yml`, where `{project}` corresponds to the project as used
by ESMValTool in :ref:`Datasets` and "something" is arbitrary.

Format of the extra facets files
--------------------------------
The extra facets are given in a yaml file, whose file name identifies the
project. Inside the file there is a hierarchy of nested dictionaries with the
following levels. At the top there is the `dataset` facet, followed by the `mip`
table, and finally the `short_name`. The leaf dictionary placed here gives the
extra facets that will be made available to data finder and the fix
infrastructure. The following example illustrates the concept.

.. _extra-facets-example-1:

.. code-block:: yaml
   :caption: Extra facet example file `native6-era5.yml`

   ERA5:
     Amon:
       tas: {source_var_name: "t2m", cds_var_name: "2m_temperature"}

The three levels of keys in this mapping can contain
`Unix shell-style wildcards <https://en.wikipedia.org/wiki/Glob_(programming)#Syntax>`_.
The special characters used in shell-style wildcards are:

+------------+----------------------------------------+
|Pattern     | Meaning                                |
+============+========================================+
| ``*``      |   matches everything                   |
+------------+----------------------------------------+
| ``?``      |   matches any single character         |
+------------+----------------------------------------+
| ``[seq]``  |   matches any character in ``seq``     |
+------------+----------------------------------------+
| ``[!seq]`` |   matches any character not in ``seq`` |
+------------+----------------------------------------+

where ``seq`` can either be a sequence of characters or just a bunch of characters,
for example ``[A-C]`` matches the characters ``A``, ``B``, and ``C``,
while ``[AC]`` matches the characters ``A`` and ``C``.

For example, this is used to automatically add ``product: output1`` to any
variable of any CMIP5 dataset that does not have a ``product`` key yet:

.. code-block:: yaml
   :caption: Extra facet example file `cmip5-product.yml <https://github.com/ESMValGroup/ESMValCore/blob/main/esmvalcore/_config/extra_facets/cmip5-product.yml>`_

   '*':
     '*':
       '*': {product: output1}

Location of the extra facets files
----------------------------------
Extra facets files can be placed in several different places. When we use them
to support a particular use-case within the ESMValTool project, they will be
provided in the sub-folder `extra_facets` inside the package
`esmvalcore._config`. If they are used from the user side, they can be either
placed in `~/.esmvaltool/extra_facets` or in any other directory of the users
choosing. In that case this directory must be added to the `config-user.yml`
file under the `extra_facets_dir` setting, which can take a single directory or
a list of directories.

The order in which the directories are searched is

1. The internal directory `esmvalcore._config/extra_facets`
2. The default user directory `~/.esmvaltool/extra_facets`
3. The custom user directories in the order in which they are given in
   `config-user.yml`.

The extra facets files within each of these directories are processed in
lexicographical order according to their file name.

In all cases it is allowed to supersede information from earlier files in later
files. This makes it possible for the user to effectively override even internal
default facets, for example to deal with local particularities in the data
handling.

Use of extra facets
-------------------
For extra facets to be useful, the information that they provide must be
applied. There are fundamentally two places where this comes into play. One is
:ref:`the datafinder<extra-facets-data-finder>`, the other are
:ref:`fixes<extra-facets-fixes>`.
