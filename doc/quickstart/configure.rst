.. _config:

*************
Configuration
*************

.. _config_overview:

Overview
========

Similar to `Dask <https://docs.dask.org/en/stable/configuration.html>`__,
ESMValCore provides one single configuration object that consists of a single
nested dictionary for its configuration.

.. note::

  In v2.12.0, a redesign process of ESMValTool/Core's configuration started.
  Its main aim is to simplify the configuration by moving from many different
  configuration files for individual components to one configuration object
  that consists of a single nested dictionary (similar to `Dask's configuration
  <https://docs.dask.org/en/stable/configuration.html>`__).
  This change will not be implemented in one large pull request but rather in a
  step-by-step procedure.
  Thus, the configuration might appear inconsistent until this redesign is
  finished.
  A detailed plan for this new configuration is outlined in :issue:`2371`.


.. _config_for_cli:

Specify configuration for ``esmvaltool`` command line tool
==========================================================

When running recipes via the :ref:`command line <running>`, configuration
options can be specified via YAML files and command line arguments.
The options from all YAML files and command line arguments are merged together
using :func:`dask.config.collect` to create a single configuration object,
which properly considers nested objects (see :func:`dask.config.update` for
details).
Configuration options given via the command line will always be preferred over
options given via YAML files.


.. _config_yaml_files:

YAML files
----------

:ref:`Configuration options <config_options>` can be specified via YAML files
(i.e., ``*.yaml`` and ``*.yml``).

A file could look like this (for example, located at
``~/.config/esmvaltool/config.yml``):

.. code-block:: yaml

  output_dir: ~/esmvaltool_output
  max_parallel_tasks: 1

ESMValCore searches for **all** YAML files in **each** of the following
locations and merges them together:

1. The directory specified via the ``--config_dir`` command line argument.

2. The user configuration directory: by default ``~/.config/esmvaltool``, but
   this location can be changed with the ``ESMVALTOOL_CONFIG_DIR`` environment
   variable.

Preference follows the order in the list above (i.e., the directory specified
via command line argument is preferred over the user configuration directory).
Within a directory, files are sorted lexicographically, and later files (e.g.,
``z.yml``) will take precedence over earlier files (e.g., ``a.yml``).

.. warning::

  ESMValCore will read **all** YAML files in these configuration directories.
  Thus, other YAML files in this directory which are not valid configuration
  files (like the old ``config-developer.yml`` files) will lead to errors.
  Make sure to move these files to a different directory.

The minimal required configuration for the tool is that you configure where
it can find :ref:`input data <config-data-sources>`. In addition to that, you
may copy the default configuration file with :ref:`top level options <config_options>`

To get a copy of the default configuration file, you can run the command:

.. code-block:: bash

  esmvaltool config copy defaults/config-user.yml

This will copy the file to your configuration directory and you can tailor it
for your system, e.g. set the ``output_dir`` to a path where ESMValTool can
store its output files.

Command line arguments
----------------------

All :ref:`configuration options <config_options>` can also be given as command
line arguments to the ``esmvaltool`` executable.

Example:

.. code-block:: bash

  esmvaltool run --max_parallel_tasks=2 /path/to/recipe.yml

Options given via command line arguments will always take precedence over
options specified via YAML files.


.. _config_for_api:

Specify/access configuration for Python API
===========================================

When running recipes with the :ref:`experimental Python API
<experimental_api>`, configuration options can be specified and accessed via
the :py:data:`~esmvalcore.config.CFG` object.
For example:

.. code-block:: python

  >>> from esmvalcore.config import CFG
  >>> CFG['output_dir'] = '~/esmvaltool_output'
  >>> CFG['output_dir']
  PosixPath('/home/user/esmvaltool_output')

Or, alternatively, via a context manager:

.. code-block:: python

  >>> with CFG.context(log_level="debug"):
  ...     print(CFG["log_level"])
  debug
  >>> print(CFG["log_level"])
  info

This will also consider YAML configuration files in the user configuration
directory (by default ``~/.config/esmvaltool``, but this can be changed with
the ``ESMVALTOOL_CONFIG_DIR`` environment variable).

More information about this can be found :ref:`here <api_configuration>`.


.. _config_options:

Top level configuration options
===============================

Note: the following entries use Python syntax.
For example, Python's ``None`` is YAML's ``null``, Python's ``True`` is YAML's
``true``, and Python's ``False`` is YAML's ``false``.

.. list-table::
   :widths: 15 50 15 20
   :header-rows: 1

   * - Option
     - Description
     - Type
     - Default value
   * - ``auxiliary_data_dir``
     - Directory where auxiliary data is stored. [#f1]_
     - :obj:`str`
     - ``~/auxiliary_data``
   * - ``check_level``
     - Sensitivity of the CMOR check (``debug``, ``strict``, ``default``, ``relaxed``, ``ignore``), see :ref:`cmor_check_strictness`.
     - :obj:`str`
     - ``default``
   * - ``compress_netcdf``
     - Use netCDF compression.
     - :obj:`bool`
     - ``False``
   * - ``config_developer_file``
     - Path to custom :ref:`config-developer`.
     - :obj:`str`
     - ``None`` (default file)
   * - ``dask``
     - :ref:`config-dask`.
     - :obj:`dict`
     - See :ref:`config-dask-defaults`
   * - ``diagnostics``
     - Only run the selected diagnostics from the recipe, see :ref:`running`.
     - :obj:`list` or :obj:`str`
     - ``None`` (all diagnostics)
   * - ``download_dir``
     - [deprecated] Directory where downloaded data will be stored. [#f2]_
     - :obj:`str`
     - ``~/climate_data``
   * - ``drs``
     - [deprecated] Directory structure for input data. [#f2]_
     - :obj:`dict`
     - ``{CMIP3: ESGF, CMIP5: ESGF, CMIP6: ESGF, CORDEX: ESGF, obs4MIPs: ESGF}``
   * - ``exit_on_warning``
     - Exit on warning (only used in NCL diagnostic scripts).
     - :obj:`bool`
     - ``False``
   * - ``log_level``
     - Log level of the console (``debug``, ``info``, ``warning``, ``error``).
     - :obj:`str`
     - ``info``
   * - ``logging``
     - :ref:`config-logging`.
     - :obj:`dict`
     - See :ref:`config-logging`
   * - ``max_datasets``
     - Maximum number of datasets to use, see :ref:`running`.
     - :obj:`int`
     - ``None`` (all datasets from recipe)
   * - ``max_parallel_tasks``
     - Maximum number of parallel processes, see :ref:`task_priority`. [#f5]_
     - :obj:`int`
     - ``None`` (number of available CPUs)
   * - ``max_years``
     - Maximum number of years to use, see :ref:`running`.
     - :obj:`int`
     - ``None`` (all years from recipe)
   * - ``output_dir``
     - Directory where all output will be written, see :ref:`outputdata`.
     - :obj:`str`
     - ``~/esmvaltool_output``
   * - ``output_file_type``
     - Plot file type.
     - :obj:`str`
     - ``png``
   * - ``profile_diagnostic``
     - Use a profiling tool for the diagnostic run. [#f3]_
     - :obj:`bool`
     - ``False``
   * - ``projects``
     - :ref:`config-projects`.
     - :obj:`dict`
     - See table in :ref:`config-projects`
   * - ``remove_preproc_dir``
     - Remove the ``preproc`` directory if the run was successful, see :ref:`preprocessed_datasets`.
     - :obj:`bool`
     - ``True``
   * - ``resume_from``
     - Resume previous run(s) by using preprocessor output files from these output directories, see :ref:`running`.
     - :obj:`list` of :obj:`str`
     - ``[]``
   * - ``rootpath``
     - [deprecated] Rootpaths to the data from different projects. [#f2]_
     - :obj:`dict`
     - ``{default: ~/climate_data}``
   * - ``run_diagnostic``
     - Run diagnostic scripts, see :ref:`running`.
     - :obj:`bool`
     - ``True``
   * - ``save_intermediary_cubes``
     - Save intermediary cubes from the preprocessor, see also :ref:`preprocessed_datasets`.
     - :obj:`bool`
     - ``False``
   * - ``search_data``
     - Perform a quick or complete search for input data. When set to ``quick``,
       search will stop as soon as a result is found. :ref:`Data sources <config-data-sources>`
       with a lower value for ``priority`` will be searched first. (``quick``, ``complete``)
     - :obj:`str`
     - ``quick``
   * - ``search_esgf``
     - [deprecated] Automatic data download from ESGF (``never``, ``when_missing``, ``always``). [#f2]_
     - :obj:`str`
     - ``never``
   * - ``skip_nonexistent``
     - Skip non-existent datasets, see :ref:`running`.
     - :obj:`bool`
     - ``False``

.. [#f1] The ``auxiliary_data_dir`` setting is the path to place any required
    additional auxiliary data files.
    This is necessary because certain Python toolkits, such as cartopy, will
    attempt to download data files at run time, typically geographic data files
    such as coastlines or land surface maps.
    This can fail if the machine does not have access to the wider internet.
    This location allows the user to specify where to find such files if they
    can not be downloaded at runtime.
    The example configuration file already contains two valid locations for
    ``auxiliary_data_dir`` directories on CEDA-JASMIN and DKRZ, and a number of
    such maps and shapefiles (used by current diagnostics) are already there.
    You will need ``esmeval`` group workspace membership to access the JASMIN
    one (see `instructions
    <https://help.jasmin.ac.uk/article/199-introduction-to-group-workspaces>`_
    how to gain access to the group workspace.

    .. warning::

       This setting is not for model or observational datasets, rather it is
       for extra data files such as shapefiles or other data sources needed by
       the diagnostics.
.. [#f2] This option is scheduled for removal in v2.14.0. Please use
    :ref:`data sources <config-data-sources>` to configure data finding instead.
.. [#f3] The ``profile_diagnostic`` setting triggers profiling of Python
    diagnostics, this will tell you which functions in the diagnostic took most
    time to run.
    For this purpose we use `vprof <https://github.com/nvdv/vprof>`_.
    For each diagnostic script in the recipe, the profiler writes a ``.json``
    file that can be used to plot a `flame graph
    <https://queue.acm.org/detail.cfm?id=2927301>`__ of the profiling
    information by running

    .. code-block:: bash

      vprof --input-file esmvaltool_output/recipe_output/run/diagnostic/script/profile.json

    Note that it is also possible to use vprof to understand other resources
    used while running the diagnostic, including execution time of different
    code blocks and memory usage.
.. [#f5] When using ``max_parallel_tasks`` with a value larger than 1 with the
   Dask threaded scheduler, every task will start ``num_workers`` threads.
   To avoid running out of memory or slowing down computations due to competition
   for resources, it is recommended to set ``num_workers`` such that
   ``max_parallel_tasks * num_workers`` approximately equals the number of CPU cores.
   The number of available CPU cores can be found by running
   ``python -c 'import os; print(len(os.sched_getaffinity(0)))'``.
   See :ref:`config-dask-threaded-scheduler` for information on how to configure
   ``num_workers``.


.. _config-dask:

Dask configuration
==================

Configure Dask in the ``dask`` section.

The :ref:`preprocessor functions <preprocessor_functions>` and many of the
:ref:`Python diagnostics in ESMValTool <esmvaltool:recipes>` make use of the
:ref:`Iris <iris:iris_docs>` library to work with the data.
In Iris, data can be either :ref:`real or lazy <iris:real_and_lazy_data>`.
Lazy data is represented by `dask arrays <https://docs.dask.org/en/stable/array.html>`__.
Dask arrays consist of many small
`numpy arrays <https://numpy.org/doc/stable/user/absolute_beginners.html#what-is-an-array>`__
(called chunks) and if possible, computations are run on those small arrays in
parallel.
In order to figure out what needs to be computed when, Dask makes use of a
'`scheduler <https://docs.dask.org/en/stable/scheduling.html>`__'.
The default (thread-based) scheduler in Dask is rather basic, so it can only
run on a single computer and it may not always find the optimal task scheduling
solution, resulting in excessive memory use when using e.g. the
:func:`esmvalcore.preprocessor.multi_model_statistics` preprocessor function.
Therefore it is recommended that you take a moment to configure the
`Dask distributed <https://distributed.dask.org>`__ scheduler.
A Dask scheduler and the 'workers' running the actual computations, are
collectively called a 'Dask cluster'.

Dask profiles
-------------

Because some recipes require more computational resources than others,
ESMValCore provides the option to define "Dask profiles".
These profiles can be used to update the `Dask user configuration
<https://docs.dask.org/en/stable/configuration.html>`__ per recipe run.
The Dask profile can be selected in a YAML configuration file via

.. code:: yaml

  dask:
    use: <NAME_OF_PROFILE>

or alternatively in the command line via

.. code:: bash

  esmvaltool run --dask='{"use": "<NAME_OF_PROFILE>"}' recipe_example.yml

Available predefined Dask profiles:

- ``local_threaded`` (selected by default): use `threaded scheduler
  <https://docs.dask.org/en/stable/scheduling.html#local-threads>`__ without
  any further options.
- ``local_distributed``: use `local distributed scheduler
  <https://docs.dask.org/en/stable/scheduling.html#dask-distributed-local>`__
  without any further options.
- ``debug``: use `synchronous Dask scheduler
  <https://docs.dask.org/en/stable/scheduling.html#single-thread>`__ for
  debugging purposes.
  Best used with ``max_parallel_tasks: 1``.

To copy these predefined profiles to your configuration directory for further
customization, run the command:

.. code:: bash

  esmvaltool config copy defaults/dask.yml

Dask distributed scheduler configuration
----------------------------------------

Here, some examples are provided on how to use a custom Dask distributed
scheduler.
Extensive documentation on setting up Dask Clusters is available `here
<https://docs.dask.org/en/latest/deploying.html>`__.

.. note::

  If not all preprocessor functions support lazy data, computational
  performance may be best with the :ref:`threaded scheduler
  <config-dask-threaded-scheduler>`.
  See :issue:`674` for progress on making all preprocessor functions lazy.

*Personal computer*

Create a :class:`distributed.LocalCluster` on the computer running ESMValCore
using all available resources:

.. code:: yaml

  dask:
    use: local_cluster  # use "local_cluster" defined below
    profiles:
      local_cluster:
        cluster:
          type: distributed.LocalCluster

This should work well for most personal computers.

.. note::

   If running this configuration on a shared node of an HPC cluster, Dask will
   try and use as many resources it can find available, and this may lead to
   overcrowding the node by a single user (you)!

*Shared computer*

Create a :class:`distributed.LocalCluster` on the computer running ESMValCore,
with 2 workers with 2 threads/4 GiB of memory each (8 GiB in total):

.. code:: yaml

  dask:
    use: local_cluster  # use "local_cluster" defined below
    profiles:
      local_cluster:
        cluster:
          type: distributed.LocalCluster
          n_workers: 2
          threads_per_worker: 2
          memory_limit: 4GiB

this should work well for shared computers.

*Computer cluster*

Create a Dask distributed cluster on the `Levante
<https://docs.dkrz.de/doc/levante/running-jobs/index.html>`__ supercomputer
using the `Dask-Jobqueue <https://jobqueue.dask.org/en/latest/>`__ package:

.. code:: yaml

  dask:
    use: slurm_cluster  # use "slurm_cluster" defined below
    profiles:
      slurm_cluster:
        cluster:
          type: dask_jobqueue.SLURMCluster
          queue: shared
          account: <YOUR_SLURM_ACCOUNT>
          cores: 8
          memory: 7680MiB
          processes: 2
          interface: ib0
          local_directory: "/scratch/b/<YOUR_DKRZ_ACCOUNT>/dask-tmp"
          n_workers: 24

This will start 24 workers with ``cores / processes = 4`` threads each,
resulting in ``n_workers / processes = 12`` Slurm jobs, where each Slurm job
will request 8 CPU cores and 7680 MiB of memory and start ``processes = 2``
workers.
This example will use the fast infiniband network connection (called ``ib0``
on Levante) for communication between workers running on different nodes.
It is `important to set the right location for temporary storage
<https://docs.dask.org/en/latest/deploying-hpc.html#local-storage>`__, in this
case the ``/scratch`` space is used.
It is also possible to use environmental variables to configure the temporary
storage location, if you cluster provides these.

A configuration like this should work well for larger computations where it is
advantageous to use multiple nodes in a compute cluster.
See `Deploying Dask Clusters on High Performance Computers
<https://docs.dask.org/en/latest/deploying-hpc.html>`__ for more information.

*Externally managed Dask cluster*

To use an externally managed cluster, specify an ``scheduler_address`` for the
selected profile.
Such a cluster can e.g. be started using the `Dask Jupyterlab extension
<https://github.com/dask/dask-labextension#dask-jupyterlab-extension>`__:

.. code:: yaml

  dask:
    use: external  # Use the `external` profile defined below
    profiles:
      external:
        scheduler_address: "tcp://127.0.0.1:43605"

See `here <https://jobqueue.dask.org/en/latest/interactive.html>`__
for an example of how to configure this on a remote system.

For debugging purposes, it can be useful to start the cluster outside of
ESMValCore because then
`Dask dashboard <https://docs.dask.org/en/stable/dashboard.html>`__ remains
available after ESMValCore has finished running.

**Advice on choosing performant configurations**

The threads within a single worker can access the same memory locations, so
they may freely pass around chunks, while communicating a chunk between workers
is done by copying it, so this is (a bit) slower.
Therefore it is beneficial for performance to have multiple threads per worker.
However, due to limitations in the CPython implementation (known as the Global
Interpreter Lock or GIL), only a single thread in a worker can execute Python
code (this limitation does not apply to compiled code called by Python code,
e.g. numpy), therefore the best performing configurations will typically not
use much more than 10 threads per worker.

Due to limitations of the NetCDF library (it is not thread-safe), only one
of the threads in a worker can read or write to a NetCDF file at a time.
Therefore, it may be beneficial to use fewer threads per worker if the
computation is very simple and the runtime is determined by the
speed with which the data can be read from and/or written to disk.

.. _config-dask-threaded-scheduler:

Custom Dask threaded scheduler configuration
--------------------------------------------

The Dask threaded scheduler can be a good choice for recipes using a small
amount of data or when running a recipe where not all preprocessor functions
are lazy yet (see :issue:`674` for the current status).

To avoid running out of memory, it is important to set the number of workers
(threads) used by Dask to run its computations to a reasonable number.
By default, the number of CPU cores in the machine will be used, but this may
be too many on shared machines or laptops with a large number of CPU cores
compared to the amount of memory they have available.

Typically, Dask requires about 2 GiB of RAM per worker, but this may be more
depending on the computation.

To set the number of workers used by the Dask threaded scheduler, use the
following configuration:

.. code:: yaml

  dask:
    use: local_threaded  # This can be omitted
    profiles:
      local_threaded:
        num_workers: 4

.. _config-dask-defaults:

Default options
---------------

By default, the following Dask configuration is used:

.. code:: yaml

  dask:
    use: local_threaded  # use the `local_threaded` profile defined below
    profiles:
      local_threaded:
        scheduler: threads
      local_distributed:
        cluster:
          type: distributed.LocalCluster
      debug:
        scheduler: synchronous

All available options
---------------------

+-------------------------------+----------------------------------------+-----------------------------+----------------------------------------+
| Option                        | Description                            | Type                        | Default value                          |
+===============================+========================================+=============================+========================================+
| ``profiles``                  | Different Dask profiles that can be    | :obj:`dict`                 | See :ref:`config-dask-defaults`        |
|                               | selected via the ``use`` option. Each  |                             |                                        |
|                               | profile has a name (:obj:`dict` keys)  |                             |                                        |
|                               | and corresponding options (:obj:`dict` |                             |                                        |
|                               | values). See                           |                             |                                        |
|                               | :ref:`config-dask-profiles` for        |                             |                                        |
|                               | details.                               |                             |                                        |
+-------------------------------+----------------------------------------+-----------------------------+----------------------------------------+
| ``use``                       | Dask profile that is used; must be     | :obj:`str`                  | ``local_threaded``                     |
|                               | defined in the option ``profiles``.    |                             |                                        |
+-------------------------------+----------------------------------------+-----------------------------+----------------------------------------+

.. _config-dask-profiles:

Options for Dask profiles
-------------------------

+-------------------------------+----------------------------------------+-----------------------------+----------------------------------------+
| Option                        | Description                            | Type                        | Default value                          |
+===============================+========================================+=============================+========================================+
| ``cluster``                   | Keyword arguments to initialize a Dask | :obj:`dict`                 | If omitted, use externally managed     |
|                               | distributed cluster. Needs the option  |                             | cluster if ``scheduler_address`` is    |
|                               | ``type``, which specifies the class of |                             | given or a :ref:`Dask threaded         |
|                               | the cluster. The remaining options are |                             | scheduler                              |
|                               | passed as keyword arguments to         |                             | <config-dask-threaded-scheduler>`      |
|                               | initialize that class. Cannot be used  |                             | otherwise.                             |
|                               | in combination with                    |                             |                                        |
|                               | ``scheduler_address``.                 |                             |                                        |
+-------------------------------+----------------------------------------+-----------------------------+----------------------------------------+
| ``scheduler_address``         | Scheduler address of an externally     | :obj:`str`                  | If omitted, use a Dask distributed     |
|                               | managed cluster. Will be passed to     |                             | cluster if ``cluster`` is given or a   |
|                               | :class:`distributed.Client`. Cannot be |                             | :ref:`Dask threaded scheduler          |
|                               | used in combination with ``cluster``.  |                             | <config-dask-threaded-scheduler>`      |
|                               |                                        |                             | otherwise.                             |
+-------------------------------+----------------------------------------+-----------------------------+----------------------------------------+
| All other options             | Passed as keyword arguments to         | Any                         | No defaults.                           |
|                               | :func:`dask.config.set`.               |                             |                                        |
+-------------------------------+----------------------------------------+-----------------------------+----------------------------------------+


.. _config-logging:

Logging configuration
=====================

Configure what information is logged and how it is presented in the ``logging``
section.

.. note::

   Not all logging configuration is available here yet, see :issue:`2596`.

Configuration file example:

.. code:: yaml

   logging:
     log_progress_interval: 10s

will log progress of Dask computations every 10 seconds instead of showing a
progress bar.

Command line example:

.. code:: bash

   esmvaltool run --logging='{"log_progress_interval": "1m"}' recipe_example.yml


will log progress of Dask computations every minute instead of showing a
progress bar.

Available options:

+-------------------------------+----------------------------------------+-----------------------------+----------------------------------------+
| Option                        | Description                            | Type                        | Default value                          |
+===============================+========================================+=============================+========================================+
| ``log_progress_interval``     | When running computations with Dask,   | :obj:`str` or :obj:`float`  | 0                                      |
|                               | log progress every                     |                             |                                        |
|                               | ``log_progress_interval`` instead of   |                             |                                        |
|                               | showing a progress bar. The value can  |                             |                                        |
|                               | be specified in the format accepted by |                             |                                        |
|                               | :func:`dask.utils.parse_timedelta`. A  |                             |                                        |
|                               | negative value disables any progress   |                             |                                        |
|                               | reporting. A progress bar is only      |                             |                                        |
|                               | shown if ``max_parallel_tasks: 1``.    |                             |                                        |
+-------------------------------+----------------------------------------+-----------------------------+----------------------------------------+


.. _config-projects:

Project-specific configuration
==============================

Configure project-specific settings in the ``projects`` section.

Top-level keys in this section are projects, e.g., ``CMIP6``, ``CORDEX``, or
``obs4MIPs``.

Example:

.. code-block:: yaml

  projects:
    CMIP6:
      ...  # project-specific options

The following project-specific options are available:

.. list-table::
   :widths: 15 50 15 20
   :header-rows: 1

   * - Option
     - Description
     - Type
     - Default value
   * - ``cmor_table``
     - :ref:`CMOR tables <cmor_tables>` are used to define the variables that ESMValCore can work with. Refer to :ref:`cmor_table_configuration` for available options.
     - :obj:`dict`
     - ``{}``
   * - ``data``
     - Data sources are used to find input data and have to be configured before running the tool. Refer to :ref:`config-data-sources` for details.
     - :obj:`dict`
     - ``{}``
   * - ``extra_facets``
     - Extra key-value pairs ("*facets*") added to datasets in addition to the facets defined in the recipe. Refer to  :ref:`config-extra-facets` for details.
     - :obj:`dict`
     - Refer to :ref:`config-extra-facets-defaults`.
   * - ``preprocessor_filename_template``
     - A template defining the filenames to use for :ref:`preprocessed data <preprocessed_datasets>` when running a :ref:`recipe <recipe>`. Refer to  :ref:`config-preprocessor-filename-template` for details.
     - :obj:`str`
     - Refer to :ref:`config-preprocessor-filename-template`.

.. _cmor_table_configuration:

CMOR table configuration
------------------------

:ref:`CMOR tables <cmor_tables>` are used to define the variables that ESMValCore
can work with.

Default values are provided in ``defaults/cmor_tables.yml``,
for example:

.. literalinclude:: ../configurations/defaults/cmor_tables.yml
    :language: yaml
    :caption: First few lines of ``defaults/cmor_tables.yml``
    :end-before: CMIP3:

The ``type`` parameter defines which class is used to read the CMOR tables for a
project, it should be a subclass of :class:`esmvalcore.cmor.table.InfoBase`.
The other parameters are passed as keyword arguments to the class when it is
created. See :mod:`esmvalcore.cmor.table` for a description of the built-in
classes for reading CMOR tables and their parameters.

Most users will not need to change the CMOR table configuration. However, if you
have variables you would like to use which are not in the standard tables,
it is recommended that you start from the default configuration and extend the
list of ``paths`` with the directory where your custom CMOR tables are stored.

If you have data that is not described in a CMOR table at all, you can use
the :class:`esmvalcore.cmor.table.NoInfo` class to indicate that no CMOR table
is available. In that case you will need to provide all necessary facets
for finding and saving the data in the :ref:`recipe <recipe>` or
:class:`~esmvalcore.dataset.Dataset`, and
:ref:`CMOR checks <cmor_check_strictness>` will be skipped.

.. warning::

    While it is possible to work with datasets that are not described in a CMOR
    table,the  :ref:`preprocessor functions <preprocessor>` and
    :ref:`diagnostics <esmvaltool:recipes>` have been designed to work with
    CMORized data and may not work as expected with non-CMORized data.

.. _config-data-sources:

Data sources
------------
The ``data`` section defines sources of input data. The easiest way to get
started with these is to copy one of the example configuration files and tailor
it to your needs.

To list the available example configuration files, run the command:

.. code-block:: bash

  esmvaltool config list

To use one of the example configuration files, copy it to
your configuration directory by running the command:

.. code-block:: bash

  esmvaltool config copy data-intake-esgf.yml

where ``data-intake-esgf.yml`` needs to be replaced by the name of the example
configuration you would like to use. The format of the configuration file
is described in :mod:`esmvalcore.io`.

There are three modules available as part of ESMValCore that provide data sources:

- :mod:`esmvalcore.io.intake_esgf`: Use the
  `intake-esgf <https://intake-esgf.readthedocs.io>`_ library to load data that
  is available from ESGF.
- :mod:`esmvalcore.io.local`: Use :mod:`glob` patterns to find files on a filesystem.
- :mod:`esmvalcore.io.esgf`: Use the legacy `esgf-pyclient
  <https://esgf-pyclient.readthedocs.io>`_ library to find and download data
  from ESGF.

Adding a custom data source is relatively easy and is explained in
:mod:`esmvalcore.io.protocol`.

There are various use cases and we provide example configurations for each of
them below.

Personal computer
`````````````````

On a personal computer, the recommended setup can be obtained by running the
commands:

.. code-block:: bash

    esmvaltool config copy data-intake-esgf.yml
    esmvaltool config copy data-local-esmvaltool.yml

This will use the :mod:`esmvalcore.io.intake_esgf` module to access data
that is available through ESGF and use :mod:`esmvalcore.io.local` to find
observational and reanalysis datasets that have been
:ref:`CMORized with ESMValTool <esmvaltool:inputdata_observations>`
(``OBS6`` and ``OBS`` projects for CMIP6- and CMIP5-style CMORization
respectively) or are supported in their :ref:`native format <read_native_datasets>`
through the ``native6`` project.

.. warning::

    It is important to :doc:`configure intake-esgf <intake_esgf:configure>`
    for your system before using it. Make sure to set ``local_cache`` to a path
    where it can store downloaded files, and if (some) ESGF data is already
    available on your system, point ``esg_dataroot`` to it. If you are
    missing certain search results, you may want to choose a different
    index node for searching the ESGF.

HPC system
``````````

On HPC systems, data is often stored in large shared filesystems. We have
several example configurations for popular HPC systems. To list the available
example files, run the command:

.. code-block:: bash

  esmvaltool config list data-hpc

If you are using one of the supported HPC systems, for example Jasmin, you can
copy the example configuration file by running the command:

.. code-block:: bash

  esmvaltool config copy data-hpc-badc.yml

and you should be good to go. If your HPC system is not supported yet, you can
copy one of the other example configuration files, e.g. ``data-hpc-dkrz.yml``
and tailor it for your system.

.. warning::

    It is important to :doc:`configure intake-esgf <intake_esgf:configure>`
    for your system before using it. Make sure to set ``local_cache`` to a path
    where it can store downloaded files, and if (some) ESGF data is already
    available on your system, point ``esg_dataroot`` to it. If you are
    missing certain search results, you may want to choose a different
    index node for searching the ESGF.

.. note::

    Deduplicating data found via :mod:`esmvalcore.io.intake_esgf` data sources
    and the :mod:`esmvalcore.io.local` data sources has not yet been implemented.
    Therefore it is recommended not to use the configuration option
    ``search_data: complete`` when using both data sources for the same project.
    The ``search_data: quick`` option can be safely used.

Climate model data in its native format
```````````````````````````````````````

For each of the climate models that are supported in their
native format as described in :ref:`read_native_models`, an example configuration
file is available. To list the available example files, run the command:

.. code-block:: bash

  esmvaltool config list data-native

.. _filter_load_warnings:

Filter Iris load warnings
`````````````````````````

It is possible to ignore specific warnings when loading data with Iris.
This is particularly useful for native datasets which do not follow the CMOR
standard by default and consequently produce a lot of warnings when handled by
Iris.
This can be configured using the ``ignore_warnings`` argument to
:class:`esmvalcore.io.local.LocalDataSource`.

Here is an example on how to ignore specific warnings when loading data from
the ``EMAC`` model in its native format:

.. literalinclude:: ../configurations/data-native-emac.yml
    :language: yaml

The keyword arguments specified in the list items are directly passed to
:func:`warnings.filterwarnings` in addition to ``action=ignore``.

.. _config-extra-facets:

Extra Facets
------------

It can be useful to automatically add extra :ref:`facets <facets>` to variables
or datasets without explicitly specifying them in the recipe.
These facets can be used for :ref:`finding data <extra-facets-data-finder>` or
for providing extra information to the functions that
:ref:`fix data <extra-facets-fixes>` before passing it on to the preprocessor.

To support this, we provide the **extra facets** facilities.
Facets are the key-value pairs described in :ref:`Datasets`.
Extra facets allows for the addition of more details per project, dataset, MIP
table, and variable name.

Format of the extra facets
``````````````````````````

Extra facets are configured in the ``extra_facets`` section of the
project-specific configuration.
They are specified in nested dictionaries with the following levels:

1. Dataset name
2. MIP table
3. Variable short name

Example:

.. code-block:: yaml

  projects:
    CMIP6:
      extra_facets:
        CanESM5:  # dataset name
          Amon:  # MIP table
            tas:  # variable short name
              a_new_key: a_new_value  # extra facets

The three top levels under ``extra_facets`` (dataset name, MIP table, and
variable short name) can contain `Unix shell-style wildcards
<https://en.wikipedia.org/wiki/Glob_(programming)#Syntax>`_.
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

where ``seq`` can either be a sequence of characters or just a bunch of
characters, for example ``[A-C]`` matches the characters ``A``, ``B``, and
``C``, while ``[AC]`` matches the characters ``A`` and ``C``.

Examples:

.. code-block:: yaml

  projects:
    CMIP6:
      extra_facets:
        CanESM5:  # dataset name
          "*":  # MIP table
            "*":  # variable short name
              a_new_key: a_new_value  # extra facets

Here, the extra facet ``a_new_key: a_new_value`` will be added to any *CMIP6*
data from model *CanESM5*.

If keys are duplicated, later keys will take precedence over earlier keys:

.. code-block:: yaml

  projects:
    CMIP6:
      extra_facets:
        CanESM5:
          "*":
            "*":
              shared_key: with_wildcard
              unique_key_1: test
          Amon:
            tas:
              shared_key: without_wildcard
              unique_key_2: test

Here, the following extra facets will be added to a dataset with project
*CMIP6*, name *CanESM5*, MIP table *Amon*, and variable short name *tas*:

.. code-block:: yaml

  unique_key_1: test
  shared_key: without_wildcard  # takes value from later entry
  unique_key_2: test

.. _config-extra-facets-defaults:

Default extra facets
````````````````````

Default extra facets are specified in ``extra_facets_*.yml`` files located in
`this
<https://github.com/ESMValGroup/ESMValCore/tree/main/esmvalcore/config/configurations/defaults>`__
directory.

.. _config-preprocessor-filename-template:

Preprocessor output filenames
-----------------------------

The filename to use for saving :ref:`preprocessed data <preprocessed_datasets>`
when running a :ref:`recipe <recipe>` is configured using ``preprocessor_filename_template``,
similar to the filename template in :class:`esmvalcore.io.local.LocalDataSource`.

Default values are provided in ``defaults/preprocessor_filename_template.yml``,
for example:

.. literalinclude:: ../configurations/defaults/preprocessor_filename_template.yml
    :language: yaml
    :caption: First few lines of ``defaults/preprocessor_filename_template.yml``
    :end-before: # Observational

The facet names from the template are replaced with the facet values from the
recipe to create a filename. The extension ``.nc`` (and if applicable, a start
and end time) will automatically be appended to the filename.

If no ``preprocessor_filename_template`` is configured for a project, the facets
describing the dataset in the recipe, as stored in
:attr:`esmvalcore.dataset.Dataset.minimal_facets`, are used.

.. _config-esgf:

ESGF configuration
==================

The ``esmvaltool run`` command can automatically download the files required
to run a recipe from ESGF for the projects CMIP3, CMIP5, CMIP6, CORDEX, and obs4MIPs.

Refer to :ref:`config-data-sources` for instructions on how to set this up. This
section describes additional configuration options for the :mod:`esmvalcore.io.esgf`
module, which is based on the legacy esgf-pyclient_ library. Most users
will not need this.

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

.. _config_esgf_pyclient:

Configuration file
------------------
An optional configuration file can be created for configuring how the
:class:`esmvalcore.io.esgf.ESGFDataSource` uses esgf-pyclient_
to find and download data.
The name of this file is ``~/.esmvaltool/esgf-pyclient.yml``.

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

    search_connection:
      urls:
        - 'https://esgf-node.ornl.gov/esgf-1-5-bridge'
        - 'https://esgf.ceda.ac.uk/esg-search'
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

Note that by default the tool will try searching the
`ESGF index nodes <https://esgf.llnl.gov/nodes.html>`_
in the order provided in the configuration file and use the first one that is
online.
Some ESGF index nodes may return search results faster than others, so you may
be able to speed up the search for files by experimenting with placing different
index nodes at the top of the list.

.. warning::

   ESGF is currently
   `transitioning to new server technology <https://github.com/ESGF/esgf-roadmap/blob/main/status/README.md>`__
   and all of the above indices are expected to go offline except the first one.

Issues with https://esgf-node.ornl.gov/esgf-1-5-bridge can be reported
`here <https://github.com/esgf2-us/esg_fastapi/issues>`__.

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

.. deprecated:: 2.14.0

  The developer configuration file is deprecated and will no longer be supported
  in v2.16.0. Please use the :ref:`project-specific configuration <config-projects>`
  instead.

  See the `v2.13.0 <https://docs.esmvaltool.org/projects/ESMValCore/en/v2.13.0/quickstart/configure.html#developer-configuration-file>`__
  documentation for previous usage of the developer configuration file.

.. warning::

    Make sure that **no** ``config-developer.yml`` file is saved
    in the ESMValCore configuration directories (see
    :ref:`config_overview` for details), as it does not contain configuration
    options that are valid in the new configuration system.

Upgrade instructions for finding files
--------------------------------------

The ``input_dir``, ``input_file``, and ``ignore_warnings`` settings have
been replaced by the :class:`esmvalcore.io.local.LocalDataSource`, which can be
configured via :ref:`data sources <config-data-sources>`.

Example 1: A config-developer.yml file specifying a directory structure for
CMIP6 data:

.. code:: yaml

    CMIP6:
      input_dir:
        ESGF: "{project}/{activity}/{institute}/{dataset}/{exp}/{ensemble}/{mip}/{short_name}/{grid}/{version}"
      input_file: "{short_name}_{mip}_{dataset}_{exp}_{ensemble}_{grid}*.nc"

and associated ``rootpath`` and ``drs`` settings:

.. code:: yaml

    rootpath:
      CMIP6: ~/climate_data
    drs:
      CMIP6: ESGF

would translate to the following new configuration:

.. code:: yaml

    projects:
      CMIP6:
        data:
          local:
            type: esmvalcore.io.local.LocalDataSource
            rootpath: ~/climate_data
            dirname_template: "{project}/{activity}/{institute}/{dataset}/{exp}/{ensemble}/{mip}/{short_name}/{grid}/{version}"
            filename_template: "{short_name}_{mip}_{dataset}_{exp}_{ensemble}_{grid}*.nc"

Upgrade instructions for naming preprocessor output files
---------------------------------------------------------

The ``output_file`` setting has been replaced by the
``preprocessor_filename_template`` settings described in
:ref:`config-preprocessor-filename-template`.

Example 1: A config-developer.yml file specifying preprocessor output filenames
for CMIP6 data:

.. code:: yaml

    CMIP6:
      output_file: "{project}_{dataset}_{mip}_{exp}_{ensemble}_{short_name}_{grid}"


would translate to the following new configuration:

.. code:: yaml

    projects:
      CMIP6:
        preprocessor_filename_template: "{project}_{dataset}_{mip}_{exp}_{ensemble}_{short_name}_{grid}"

Upgrade instructions for using custom CMOR tables
-------------------------------------------------

The CMOR tables can now be configured via :ref:`cmor_table_configuration`. The
following mapping applies:

- ``cmor_type`` has been replaced by ``type``
- ``cmor_strict`` has been replaced by ``strict``
- ``cmor_path`` has been replaced by ``paths``
- ``cmor_default_table_prefix`` is no longer needed.

Because it is now possible to configure multiple paths to directories containing
CMOR tables per project, the ``custom`` project for specifying additional custom
CMOR tables is no longer needed or supported.

Example 1: A config-developer.yml file specifying different CMIP6 CMOR tables
than the default ones, augmented by the default custom CMOR tables:

.. code-block:: yaml

  CMIP6:
    cmor_path: /path/to/cmip6-cmor-tables
    cmor_strict: true
    cmor_type: CMIP6

would translate to the following new configuration:

.. code-block:: yaml

  projects:
    CMIP6:
      cmor_table:
        type: esmvalcore.cmor.table.CMIP6Info
        strict: true
        paths:
          - /path/to/cmip6-cmor-tables
          - cmip6-custom

where the ``cmip6-custom`` relative path refers to
`esmvalcore/cmor/tables/cmip6-custom <https://github.com/ESMValGroup/ESMValCore/tree/main/esmvalcore/cmor/tables/cmip6-custom>`__
and ``type`` refers to :class:`esmvalcore.cmor.table.CMIP6Info`.

Example 2: A config-developer.yml file specifying additional custom CMOR tables:

.. code:: yaml

  CMIP6:
    cmor_path: cmip6
    cmor_strict: true
    cmor_type: CMIP6
  custom:
    cmor_path: /path/to/custom-cmip5-style-tables

would translate to the following new configuration:

.. code:: yaml

  projects:
    CMIP6:
      cmor_table:
        type: esmvalcore.cmor.table.CMIP6Info
        strict: true
        paths:
          - cmip6
          - cmip6-custom
          - /path/to/custom-cmip6-style-tables

where the relative paths ``cmip6`` and ``cmip6-custom`` refer to
`esmvalcore/cmor/tables/cmip6 <https://github.com/ESMValGroup/ESMValCore/tree/main/esmvalcore/cmor/tables/cmip6>`__
and
`esmvalcore/cmor/tables/cmip6-custom <https://github.com/ESMValGroup/ESMValCore/tree/main/esmvalcore/cmor/tables/cmip6-custom>`__
respectively and the directory ``/path/to/custom-cmip6-style-tables`` contains
the additional custom CMOR tables in CMIP6 format. A script to translate custom
CMIP5 format tables to CMIP6 format tables is available
`here <https://github.com/ESMValGroup/ESMValCore/blob/main/esmvalcore/cmor/tables/cmip6-custom/convert-cmip5-to-cmip6.py>`__.
Note that it is no longer possible to mix CMIP6-style tables with custom
CMIP5-style tables for the same project.

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
