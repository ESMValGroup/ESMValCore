.. _experimental_api:

Experimental API
================

This page describes the new ESMValCore API.
The API module is available in the submodule ``esmvalcore.experimental``.
The API is under development, so use at your own risk!

Config
******

Configuration of ESMValCore/Tool is done via the ``Config`` object.
The global configuration can be imported from the ``esmvalcore.experimental`` module as ``CFG``:

.. code-block:: python

    >>> from esmvalcore.experimental import CFG
    >>> CFG
    Config({'auxiliary_data_dir': PosixPath('/home/user/auxiliary_data'),
            'compress_netcdf': False,
            'config_developer_file': None,
            'config_file': PosixPath('/home/user/.esmvaltool/config-user.yml'),
            'drs': {'CMIP5': 'default', 'CMIP6': 'default'},
            'exit_on_warning': False,
            'log_level': 'info',
            'max_parallel_tasks': None,
            'output_dir': PosixPath('/home/user/esmvaltool_output'),
            'output_file_type': 'png',
            'profile_diagnostic': False,
            'remove_preproc_dir': True,
            'rootpath': {'CMIP5': '~/default_inputpath',
                         'CMIP6': '~/default_inputpath',
                         'default': '~/default_inputpath'},
            'save_intermediary_cubes': False,
            'write_netcdf': True,
            'write_plots': True})

The parameters for the user configuration file are listed `here <https://docs.esmvaltool.org/projects/ESMValCore/en/latest/quickstart/configure.html#user-configuration-file>`__.

``CFG`` is essentially a python dictionary with a few extra functions, similar to ``matplotlib.rcParams``.
This means that values can be updated like this:

.. code-block:: python

    >>> CFG['output_dir'] = '~/esmvaltool_output'
    >>> CFG['output_dir']
    PosixPath('/home/user/esmvaltool_output')

Notice that ``CFG`` automatically converts the path to an instance of ``pathlib.Path`` and expands the home directory.
All values entered into the config are validated to prevent mistakes, for example, it will warn you if you make a typo in the key:

.. code-block:: python

    >>> CFG['otoptu_dri'] = '~/esmvaltool_output'
    InvalidConfigParameter: `otoptu_dri` is not a valid config parameter.

Or, if the value entered cannot be converted to the expected type:

.. code-block:: python

    >>> CFG['max_years'] = '🐜'
    InvalidConfigParameter: Key `max_years`: Could not convert '🐜' to int

``Config`` is also flexible, so it tries to correct the type of your input if possible:

.. code-block:: python

    >>> CFG['max_years'] = '123'  # str
    >>> type(CFG['max_years'])
    int

By default, the config is loaded from the default location (``/home/user/.esmvaltool/config-user.yml``).
If it does not exist, it falls back to the default values.
to load a different file:

.. code-block:: python

    >>> CFG.load_from_file('~/my-config.yml')

Or to reload the current config:

.. code-block:: python

    >>> CFG.reload()


Session
*******

Recipes and diagnostics will be run in their own directories.
This behaviour can be controlled via the ``Session`` object.
A ``Session`` can be initiated from the global ``Config``.

.. code-block:: python

    >>> session = CFG.start_session(name='my_session')

A ``Session`` is very similar to the config.
It is also a dictionary, and copies all the keys from the ``Config``.
At this moment, ``session`` is essentially a copy of ``CFG``:

.. code-block:: python

    >>> print(session == CFG)
    True
    >>> session['output_dir'] = '~/session_output'
    >>> print(session == CFG)  # False
    False

A ``Session`` also knows about the directories where the data will stored.
The session name is used to prefix the directories.

.. code-block:: python

    >>> session.session_dir
    /home/user/esmvaltool_output/my_session_20201203_155821
    >>> session.run_dir
    /home/user/esmvaltool_output/my_session_20201203_155821/run
    >>> session.work_dir
    /home/user/esmvaltool_output/my_session_20201203_155821/work
    >>> session.preproc_dir
    /home/user/esmvaltool_output/my_session_20201203_155821/preproc
    >>> session.plot_dir
    /home/user/esmvaltool_output/my_session_20201203_155821/plots

``Session`` objects are persistent, so multiple sessions can be initiated from the ``Config``.


API reference
*************

.. autoclass:: esmvalcore.experimental.config._config_object.Config
    :no-inherited-members:
    :no-show-inheritance:

.. autoclass:: esmvalcore.experimental.config._config_object.Session
    :no-inherited-members:
    :no-show-inheritance:
