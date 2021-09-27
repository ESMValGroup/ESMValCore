.. _api_config:

Configuration
=============

This section describes the :py:class:`~esmvalcore.experimental.config` submodule of the API (:py:mod:`esmvalcore.experimental`).

Config
******

Configuration of ESMValCore/Tool is done via the :py:class:`~esmvalcore.experimental.config.Config` object.
The global configuration can be imported from the :py:mod:`esmvalcore.experimental` module as :py:data:`~esmvalcore.experimental.CFG`:

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
            'save_intermediary_cubes': False)

The parameters for the user configuration file are listed :ref:`here <user configuration file>`.

:py:data:`~esmvalcore.experimental.CFG` is essentially a python dictionary with a few extra functions, similar to :py:mod:`matplotlib.rcParams`.
This means that values can be updated like this:

.. code-block:: python

    >>> CFG['output_dir'] = '~/esmvaltool_output'
    >>> CFG['output_dir']
    PosixPath('/home/user/esmvaltool_output')

Notice that :py:data:`~esmvalcore.experimental.CFG` automatically converts the path to an instance of ``pathlib.Path`` and expands the home directory.
All values entered into the config are validated to prevent mistakes, for example, it will warn you if you make a typo in the key:

.. code-block:: python

    >>> CFG['output_directory'] = '~/esmvaltool_output'
    InvalidConfigParameter: `output_directory` is not a valid config parameter.

Or, if the value entered cannot be converted to the expected type:

.. code-block:: python

    >>> CFG['max_parallel_tasks'] = '🐜'
    InvalidConfigParameter: Key `max_parallel_tasks`: Could not convert '🐜' to int

:py:class:`~esmvalcore.experimental.config.Config` is also flexible, so it tries to correct the type of your input if possible:

.. code-block:: python

    >>> CFG['max_parallel_tasks'] = '8'  # str
    >>> type(CFG['max_parallel_tasks'])
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
This behaviour can be controlled via the :py:data:`~esmvalcore.experimental.config.Session` object.
A :py:data:`~esmvalcore.experimental.config.Session` can be initiated from the global :py:class:`~esmvalcore.experimental.config.Config`.

.. code-block:: python

    >>> session = CFG.start_session(name='my_session')

A :py:data:`~esmvalcore.experimental.config.Session` is very similar to the config.
It is also a dictionary, and copies all the keys from the :py:class:`~esmvalcore.experimental.config.Config`.
At this moment, ``session`` is essentially a copy of :py:data:`~esmvalcore.experimental.CFG`:

.. code-block:: python

    >>> print(session == CFG)
    True
    >>> session['output_dir'] = '~/my_output_dir'
    >>> print(session == CFG)  # False
    False

A :py:data:`~esmvalcore.experimental.config.Session` also knows about the directories where the data will stored.
The session name is used to prefix the directories.

.. code-block:: python

    >>> session.session_dir
    /home/user/my_output_dir/my_session_20201203_155821
    >>> session.run_dir
    /home/user/my_output_dir/my_session_20201203_155821/run
    >>> session.work_dir
    /home/user/my_output_dir/my_session_20201203_155821/work
    >>> session.preproc_dir
    /home/user/my_output_dir/my_session_20201203_155821/preproc
    >>> session.plot_dir
    /home/user/my_output_dir/my_session_20201203_155821/plots

Unlike the global configuration, of which only one can exist, multiple sessions can be initiated from :py:class:`~esmvalcore.experimental.config.Config`.


API reference
*************

.. automodule:: esmvalcore.experimental.config
    :no-inherited-members:
    :no-show-inheritance:
