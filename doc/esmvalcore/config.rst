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
* ``config-logging.yml``: stores information on logging.

and one configuration file which is distributed with ESMValTool:

* ``config-references.yml``: stores information on diagnostic and recipe authors and
  scientific journals references;

.. _user configuration file:

User configuration file
=======================

The ``config-user.yml`` is one of the two files the user needs to provide as
input arguments to the ``esmvaltool`` executable at run time, the second being
the :ref:`recipe`.

The ``config-user.yml`` configuration file contains all the global level
information needed by ESMValTool. It can be reused as many times the user needs
to before changing any of the options stored in it. This file is essentially
the gateway between the user and the machine-specific instructions to
``esmvaltool``. The following shows the default settings from the
``config-user.yml`` file with explanations in a commented line above each
option:

.. code-block:: yaml

  # Diagnostics create plots? [true]/false
  # turning it off will turn off graphical output from diagnostic
  write_plots: true

  # Diagnositcs write NetCDF files? [true]/false
  # turning it off will turn off netCDF output from diagnostic
  write_netcdf: true

  # Set the console log level debug, [info], warning, error
  # for much more information printed to screen set log_level: debug
  log_level: info
  # verbosity is deprecated and will be removed in the future
  # verbosity: 1

  # Exit on warning? true/[false]
  exit_on_warning: false

  # Plot file format? [ps]/pdf/png/eps/epsi
  output_file_type: pdf

  # Destination directory where all output will be written
  # including log files and performance stats
  output_dir: ./esmvaltool_output

  # Auxiliary data directory (used for some additional datasets)
  # this is where e.g. files can be downloaded to by a download
  # script embedded in the diagnostic
  auxiliary_data_dir: ./auxiliary_data

  # Use netCDF compression true/[false]
  compress_netcdf: false

  # Save intermediary cubes in the preprocessor true/[false]
  # set to true will save the output cube from each preprocessing step
  # these files are numbered according to the preprocessing order
  save_intermediary_cubes: false

  # Remove the preproc dir if all fine
  # if this option is set to "true", ALL preprocessor files will be removed
  # CAUTION when using: if you need those files, set it to false
  remove_preproc_dir: true

  # Run at most this many tasks in parallel null/[1]/2/3/4/..
  # Set to null to use the number of available CPUs.
  # Make sure your system has enough memory for the specified number of tasks.
  max_parallel_tasks: 1

  # Path to custom config-developer file, to customise project configurations.
  # See config-developer.yml for an example. Set to None to use the default
  config_developer_file: null

  # Get profiling information for diagnostics
  # Only available for Python diagnostics
  profile_diagnostic: false

  # Rootpaths to the data from different projects (lists are also possible)
  rootpath:
    CMIP5: [~/cmip5_inputpath1, ~/cmip5_inputpath2]
    OBS: ~/obs_inputpath
    default: ~/default_inputpath

  # Directory structure for input data: [default]/BADC/DKRZ/ETHZ/etc
  # See config-developer.yml for definitions.
  drs:
    CMIP5: default

Most of these settings are fairly self-explanatory, e.g.:

.. code-block:: yaml

  # Diagnostics create plots? [true]/false
  write_plots: true
  # Diagnositcs write NetCDF files? [true]/false
  write_netcdf: true

The ``write_plots`` setting is used to inform ESMValTool diagnostics about your
preference for creating figures. Similarly, the ``write_netcdf`` setting is a
boolean which turns on or off the writing of netCDF files by the diagnostic
scripts.

.. code-block:: yaml

  # Auxiliary data directory (used for some additional datasets)
  auxiliary_data_dir: ~/auxiliary_data

The ``auxiliary_data_dir`` setting is the path to place any required
additional auxiliary data files. This is necessary because certain
Python toolkits, such as cartopy, will attempt to download data files at run
time, typically geographic data files such as coastlines or land surface maps.
This can fail if the machine does not have access to the wider internet. This
location allows the user to specify where to find such files if they can not be
downloaded at runtime.

.. warning::

   This setting is not for model or observational datasets, rather it is for
   data files used in plotting such as coastline descriptions and so on.

A detailed explanation of the data finding-related sections of the
``config-user.yml`` (``rootpath`` and ``drs``) is presented in the
:ref:`data-retrieval` section. This section relates directly to the data
finding capabilities  of ESMValTool and are very important to be understood by
the user.

.. note::

   You choose your ``config-user.yml`` file at run time, so you could have several of
   them available with different purposes. One for a formalised run, another for
   debugging, etc.


.. _config-developer:

Developer configuration file
============================

Most users and diagnostic developers will not need to change this file,
but it may be useful to understand its content.
It will be installed along with ESMValCore and can also be viewed on GitHub:
`esmvalcore/config-developer.yml
<https://github.com/ESMValGroup/ESMValCore/blob/development/esmvalcore/config-developer.yml>`_.
This configuration file describes the file system structure and CMOR tables for several
key projects (CMIP6, CMIP5, obs4mips, OBS6, OBS) on several key machines (e.g. BADC, CP4CDS, DKRZ,
ETHZ, SMHI, BSC). CMIP data is stored as part of the Earth System Grid
Federation (ESGF) and the standards for file naming and paths to files are set
out by CMOR and DRS. For a detailed description of these standards and their
adoption in ESMValCore, we refer the user to :ref:`CMOR-DRS` section where we
relate these standards to the data retrieval mechanism of the ESMValCore.

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
ESMValCore replaces the placeholders ``[item]`` in
``input_dir`` and ``input_file`` with the values supplied in the recipe.
ESMValCore will try to automatically fill in the values for institute, frequency,
and modeling_realm based on the information provided in the CMOR tables
and/or ``config-developer.yml`` when reading the recipe. If this fails for some reason,
these values can be provided in the recipe too.

The data directory structure of the CMIP projects is set up differently
at each site. As an example, the CMIP6 directory path on BADC would be:

.. code-block:: yaml

   '{activity}/{institute}/{dataset}/{exp}/{ensemble}/{mip}/{short_name}/{grid}/{latestversion}'

The resulting directory path would look something like this:

.. code-block::

    CMIP/MOHC/HadGEM3-GC31-LL/historical/r1i1p1f3/Omon/tos/gn/latest

For a more in-depth description of how to configure ESMValCore so it can find
your data please see :ref:`CMOR-DRS`.

Preprocessor output files
-------------------------

The filename to use for preprocessed data is configured in a similar manner
using ``output_file``. Note that the extension ``.nc`` (and if applicable,
a start and end time) will automatically be appended to the filename.

CMOR table configuration
-------------------------

ESMValCore comes bundled with several CMOR tables, which are stored in the directory
`esmvalcore/cmor/tables
<https://github.com/ESMValGroup/ESMValCore/tree/development/esmvalcore/cmor/tables>`_.
These are copies of the tables available from `PCMDI <https://github.com/PCMDI>`_.

There are four settings related to CMOR tables available:

* ``cmor_type``: can be ``CMIP5`` if the CMOR table is in the same format as the
  CMIP5 table or ``CMIP6`` if the table is in the same format as the CMIP6 table.
* ``cmor_strict``: if this is set to ``false``, the CMOR table will be
  extended with variables from the ``esmvalcore/cmor/tables/custom`` directory
  and it is possible to use variables with a ``mip`` which is different from
  the MIP table in which they are defined.
* ``cmor_path``: path to the CMOR table. Defaults to the value provided in
  ``cmor_type`` written in lower case.
* ``cmor_default_table_prefix``: defaults to the value provided in ``cmor_type``.


.. _config-ref:

References configuration file
=============================

The ``config-references.yml`` file contains the list of ESMValTool diagnostic and recipe authors,
references and projects. Each author, project and reference referred to in the
documentation section of a recipe needs to be in this file in the relevant
section.

For instance, the recipe ``recipe_ocean_example.yml`` file contains the
following documentation section:

.. code-block:: yaml

  documentation
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


Logging configuration file
==========================

.. warning::
    Section to be added
