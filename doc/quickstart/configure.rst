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
  search_esgf: when_missing
  download_dir: ~/downloaded_data

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

.. deprecated:: 2.12.0

  If a single configuration file is present at its deprecated location
  ``~/.esmvaltool/config-user.yml`` or specified via the deprecated command
  line argument ``--config_file``, all potentially available new configuration
  files at ``~/.config/esmvaltool/`` and/or the location specified via
  ``--config_dir`` are ignored.
  This ensures full backwards-compatibility.
  To switch to the new configuration system outlined here, move your old
  configuration file to ``~/.config/esmvaltool/`` or to the location specified
  via ``--config_dir``, remove ``~/.esmvaltool/config-user.yml``, and omit the
  command line argument ``--config_file``.
  Alternatively, specifying the environment variable ``ESMVALTOOL_CONFIG_DIR``
  will also force the usage of the new configuration system regardless of the
  presence of any potential old configuration files.
  Support for the deprecated configuration will be removed in version 2.14.0.

To get a copy of the default configuration file, you can run

.. code-block:: bash

  esmvaltool config get_config_user --path=/target/file.yml

If the option ``--path`` is omitted, the file will be copied to
``~/.config/esmvaltool/config-user.yml``.


Command line arguments
----------------------

All :ref:`configuration options <config_options>` can also be given as command
line arguments to the ``esmvaltool`` executable.

Example:

.. code-block:: bash

  esmvaltool run --search_esgf=when_missing --max_parallel_tasks=2 /path/to/recipe.yml

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

+-------------------------------+----------------------------------------+-----------------------------+----------------------------------------+
| Option                        | Description                            | Type                        | Default value                          |
+===============================+========================================+=============================+========================================+
| ``auxiliary_data_dir``        | Directory where auxiliary data is      | :obj:`str`                  | ``~/auxiliary_data``                   |
|                               | stored. [#f1]_                         |                             |                                        |
+-------------------------------+----------------------------------------+-----------------------------+----------------------------------------+
| ``check_level``               | Sensitivity of the CMOR check          | :obj:`str`                  | ``default``                            |
|                               | (``debug``, ``strict``, ``default``    |                             |                                        |
|                               | ``relaxed``, ``ignore``), see          |                             |                                        |
|                               | :ref:`cmor_check_strictness`.          |                             |                                        |
+-------------------------------+----------------------------------------+-----------------------------+----------------------------------------+
| ``compress_netcdf``           | Use netCDF compression.                | :obj:`bool`                 | ``False``                              |
+-------------------------------+----------------------------------------+-----------------------------+----------------------------------------+
| ``config_developer_file``     | Path to custom                         | :obj:`str`                  | ``None`` (default file)                |
|                               | :ref:`config-developer`.               |                             |                                        |
+-------------------------------+----------------------------------------+-----------------------------+----------------------------------------+
| ``dask``                      | :ref:`config-dask`.                    | :obj:`dict`                 | See :ref:`config-dask-defaults`        |
+-------------------------------+----------------------------------------+-----------------------------+----------------------------------------+
| ``diagnostics``               | Only run the selected diagnostics from | :obj:`list` or :obj:`str`   | ``None`` (all diagnostics)             |
|                               | the recipe, see :ref:`running`.        |                             |                                        |
+-------------------------------+----------------------------------------+-----------------------------+----------------------------------------+
| ``download_dir``              | Directory where downloaded data will   | :obj:`str`                  | ``~/climate_data``                     |
|                               | be stored. [#f4]_                      |                             |                                        |
+-------------------------------+----------------------------------------+-----------------------------+----------------------------------------+
| ``drs``                       | Directory structure for input data.    | :obj:`dict`                 |  ``{CMIP3: ESGF, CMIP5: ESGF, CMIP6:   |
|                               | [#f2]_                                 |                             |  ESGF, CORDEX: ESGF, obs4MIPs: ESGF}`` |
+-------------------------------+----------------------------------------+-----------------------------+----------------------------------------+
| ``exit_on_warning``           | Exit on warning (only used in NCL      | :obj:`bool`                 | ``False``                              |
|                               | diagnostic scripts).                   |                             |                                        |
+-------------------------------+----------------------------------------+-----------------------------+----------------------------------------+
| ``log_level``                 | Log level of the console (``debug``,   | :obj:`str`                  | ``info``                               |
|                               | ``info``, ``warning``, ``error``).     |                             |                                        |
+-------------------------------+----------------------------------------+-----------------------------+----------------------------------------+
| ``logging``                   | :ref:`config-logging`.                 | :obj:`dict`                 | See :ref:`config-logging`              |
+-------------------------------+----------------------------------------+-----------------------------+----------------------------------------+
| ``max_datasets``              | Maximum number of datasets to use, see | :obj:`int`                  | ``None`` (all datasets from recipe)    |
|                               | :ref:`running`.                        |                             |                                        |
+-------------------------------+----------------------------------------+-----------------------------+----------------------------------------+
| ``max_parallel_tasks``        | Maximum number of parallel processes,  | :obj:`int`                  | ``None`` (number of available CPUs)    |
|                               | see :ref:`task_priority`. [#f5]_       |                             |                                        |
+-------------------------------+----------------------------------------+-----------------------------+----------------------------------------+
| ``max_years``                 | Maximum number of years to use, see    | :obj:`int`                  | ``None`` (all years from recipe)       |
|                               | :ref:`running`.                        |                             |                                        |
+-------------------------------+----------------------------------------+-----------------------------+----------------------------------------+
| ``output_dir``                | Directory where all output will be     | :obj:`str`                  | ``~/esmvaltool_output``                |
|                               | written, see :ref:`outputdata`.        |                             |                                        |
+-------------------------------+----------------------------------------+-----------------------------+----------------------------------------+
| ``output_file_type``          | Plot file type.                        | :obj:`str`                  | ``png``                                |
+-------------------------------+----------------------------------------+-----------------------------+----------------------------------------+
| ``profile_diagnostic``        | Use a profiling tool for the           | :obj:`bool`                 | ``False``                              |
|                               | diagnostic run. [#f3]_                 |                             |                                        |
+-------------------------------+----------------------------------------+-----------------------------+----------------------------------------+
| ``projects``                  | :ref:`config-projects`.                | :obj:`dict`                 | See table in :ref:`config-projects`    |
+-------------------------------+----------------------------------------+-----------------------------+----------------------------------------+
| ``remove_preproc_dir``        | Remove the ``preproc`` directory if    | :obj:`bool`                 | ``True``                               |
|                               | the run was successful, see also       |                             |                                        |
|                               | :ref:`preprocessed_datasets`.          |                             |                                        |
+-------------------------------+----------------------------------------+-----------------------------+----------------------------------------+
| ``resume_from``               | Resume previous run(s) by using        | :obj:`list` of :obj:`str`   | ``[]``                                 |
|                               | preprocessor output files from these   |                             |                                        |
|                               | output directories, see                |                             |                                        |
|                               | ref:`running`.                         |                             |                                        |
+-------------------------------+----------------------------------------+-----------------------------+----------------------------------------+
| ``rootpath``                  | Rootpaths to the data from different   | :obj:`dict`                 | ``{default: ~/climate_data}``          |
|                               | projects. [#f2]_                       |                             |                                        |
+-------------------------------+----------------------------------------+-----------------------------+----------------------------------------+
| ``run_diagnostic``            | Run diagnostic scripts, see            | :obj:`bool`                 | ``True``                               |
|                               | :ref:`running`.                        |                             |                                        |
+-------------------------------+----------------------------------------+-----------------------------+----------------------------------------+
| ``save_intermediary_cubes``   | Save intermediary cubes from the       | :obj:`bool`                 | ``False``                              |
|                               | preprocessor, see also                 |                             |                                        |
|                               | :ref:`preprocessed_datasets`.          |                             |                                        |
+-------------------------------+----------------------------------------+-----------------------------+----------------------------------------+
| ``search_esgf``               | Automatic data download from ESGF      | :obj:`str`                  | ``never``                              |
|                               | (``never``, ``when_missing``,          |                             |                                        |
|                               | ``always``). [#f4]_                    |                             |                                        |
+-------------------------------+----------------------------------------+-----------------------------+----------------------------------------+
| ``skip_nonexistent``          | Skip non-existent datasets, see        | :obj:`bool`                 | ``False``                              |
|                               | :ref:`running`.                        |                             |                                        |
+-------------------------------+----------------------------------------+-----------------------------+----------------------------------------+

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
.. [#f2] A detailed explanation of the data finding-related options ``drs``
    and ``rootpath`` is presented in the :ref:`data-retrieval` section.
    These sections relate directly to the data finding capabilities of
    ESMValCore and are very important to be understood by the user.
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
.. [#f4] The ``search_esgf`` setting can be used to disable or enable automatic
   downloads from ESGF.
   If ``search_esgf`` is set to ``never``, the tool does not download any data
   from the ESGF.
   If ``search_esgf`` is set to ``when_missing``, the tool will download any
   CMIP3, CMIP5, CMIP6, CORDEX, and obs4MIPs data that is required to run a
   recipe but not available locally and store it in ``download_dir`` using the
   ``ESGF`` directory structure defined in the :ref:`config-developer`.
   If ``search_esgf`` is set to ``always``, the tool will first check the ESGF
   for the needed data, regardless of any local data availability; if the data
   found on ESGF is newer than the local data (if any) or the user specifies a
   version of the data that is available only from the ESGF, then that data
   will be downloaded; otherwise, local data will be used.
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

+-------------------------------+----------------------------------------+-----------------------------+----------------------------------------+
| Option                        | Description                            | Type                        | Default value                          |
+===============================+========================================+=============================+========================================+
| ``extra_facets``              | Extra key-value pairs ("*facets*")     | :obj:`dict`                 | See                                    |
|                               | added to datasets in addition to the   |                             | :ref:`config-extra-facets-defaults`    |
|                               | facets defined in the recipe. See      |                             |                                        |
|                               | :ref:`config-extra-facets` for         |                             |                                        |
|                               | details.                               |                             |                                        |
+-------------------------------+----------------------------------------+-----------------------------+----------------------------------------+

.. _config-extra-facets:

Extra Facets
------------

It can be useful to automatically add extra key-value pairs to variables or
datasets without explicitly specifying them in the recipe.
These key-value pairs can be used for :ref:`finding data
<extra-facets-data-finder>` or for providing extra information to the functions
that :ref:`fix data <extra-facets-fixes>` before passing it on to the
preprocessor.

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


.. _config-esgf:

ESGF configuration
==================

The ``esmvaltool run`` command can automatically download the files required
to run a recipe from ESGF for the projects CMIP3, CMIP5, CMIP6, CORDEX, and obs4MIPs.
The downloaded files will be stored in the directory specified via the
:ref:`configuration option <config_options>` ``download_dir``.
To enable automatic downloads from ESGF, use the :ref:`configuration options
<config_options>` ``search_esgf: when_missing`` or ``search_esgf: always``.

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
An optional configuration file can be created for configuring how the tool uses
`esgf-pyclient <https://esgf-pyclient.readthedocs.io>`_
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
        .. - 'https://esgf-node.ornl.gov/esgf-1-5-bridge'
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

Users can get a copy of this file with default values by running

.. code-block:: bash

  esmvaltool config get_config_developer --path=${TARGET_FOLDER}

If the option ``--path`` is omitted, the file will be created in
``~/.esmvaltool``.

.. note::

  Remember to change the configuration option ``config_developer_file`` if you
  want to use a custom config developer file.

.. warning::

  For now, make sure that the custom ``config-developer.yml`` is **not** saved
  in the ESMValTool/Core configuration directories (see
  :ref:`config_yaml_files` for details).
  This will change in the future due to the :ref:`redesign of ESMValTool/Core's
  configuration <config_overview>`.

Example of the CMIP6 project configuration:

.. code-block:: yaml

   CMIP6:
     input_dir:
       default: '/'
       BADC: '{activity}/{institute}/{dataset}/{exp}/{ensemble}/{mip}/{short_name}/{grid}/{version}'
       DKRZ: '{activity}/{institute}/{dataset}/{exp}/{ensemble}/{mip}/{short_name}/{grid}/{version}'
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
and/or :ref:`config-extra-facets` when reading the recipe.
If this fails for some reason, these values can be provided in the recipe too.

The data directory structure of the CMIP projects is set up differently
at each site. As an example, the CMIP6 directory path on BADC would be:

.. code-block:: yaml

   '{activity}/{institute}/{dataset}/{exp}/{ensemble}/{mip}/{short_name}/{grid}/{version}'

The resulting directory path would look something like this:

.. code-block:: bash

    CMIP/MOHC/HadGEM3-GC31-LL/historical/r1i1p1f3/Omon/tos/gn/latest

Please, bear in mind that ``input_dirs`` can also be a list for those cases in
which may be needed:

.. code-block:: yaml

  - '{exp}/{ensemble}/original/{mip}/{short_name}/{grid}/{version}'
  - '{exp}/{ensemble}/computed/{mip}/{short_name}/{grid}/{version}'

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
  they are defined. Note that this option is always enabled for
  :ref:`derived variables <Variable derivation>`.
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
For :ref:`derived variables <Variable derivation>` (the ones with ``derive:
true`` in the recipe), the custom CMOR tables will always be considered.
By default, these custom tables are loaded from `esmvalcore/cmor/tables/custom
<https://github.com/ESMValGroup/ESMValCore/tree/main/esmvalcore/cmor/tables/custom>`_.
However, by using the special project ``custom`` in the
``config-developer.yml`` file with the option ``cmor_path``, a custom location
for these custom CMOR tables can be specified.
In this case, the default custom tables are extended with those entries from
the custom location (in case of duplication, the custom location tables take
precedence).

Example:

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

It is also possible to use a special coordinates file ``CMOR_coordinates.dat``,
which will extend the entries from the default one
(`esmvalcore/cmor/tables/custom/CMOR_coordinates.dat
<https://github.com/ESMValGroup/ESMValCore/tree/main/esmvalcore/cmor/tables/custom/CMOR_coordinates.dat>`_).


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
step ``load`` for all datasets of project ``EMAC`` (taken from the default
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
       default: 'Tier{tier}/{dataset}/{version}/{frequency}/{short_name}'
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
         - '{exp}/output'
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
