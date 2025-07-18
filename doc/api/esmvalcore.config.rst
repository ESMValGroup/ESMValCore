.. _api_configuration:

Configuration
=============

This section describes the :py:class:`~esmvalcore.config` module.

CFG
***

Configuration of ESMValCore/Tool is done via :py:data:`~esmvalcore.config.CFG`
object:

.. code-block:: python

    >>> from esmvalcore.config import CFG
    >>> CFG
    Config({'auxiliary_data_dir': PosixPath('/home/user/auxiliary_data'),
            'compress_netcdf': False,
            'config_developer_file': None,
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

All configuration parameters are listed :ref:`here <config_options>`.

:py:data:`~esmvalcore.config.CFG` is essentially a python dictionary with a few
extra functions, similar to :py:data:`matplotlib.rcParams`.
This means that values can be updated like this:

.. code-block:: python

    >>> CFG['output_dir'] = '~/esmvaltool_output'
    >>> CFG['output_dir']
    PosixPath('/home/user/esmvaltool_output')

Notice that :py:data:`~esmvalcore.config.CFG` automatically converts the path
to an instance of :class:`pathlib.Path` and expands the home directory.
All values entered into the config are validated to prevent mistakes, for
example, it will warn you if you make a typo in the key:

.. code-block:: python

    >>> CFG['output_directory'] = '~/esmvaltool_output'
    InvalidConfigParameter: `output_directory` is not a valid config parameter.

Or, if the value entered cannot be converted to the expected type:

.. code-block:: python

    >>> CFG['max_parallel_tasks'] = '🐜'
    InvalidConfigParameter: Key `max_parallel_tasks`: Could not convert '🐜' to int

:py:data:`~esmvalcore.config.CFG` is also flexible, so it tries to correct the
type of your input if possible:

.. code-block:: python

    >>> CFG['max_parallel_tasks'] = '8'  # str
    >>> type(CFG['max_parallel_tasks'])
    int

It is also possible to temporarily use different configuration options using
the :meth:`~esmvalcore.config.Config.context` context manager:

.. code-block:: python

    >>> with CFG.context({"output_dir": "/path/to/output"}, log_level="debug"):
    ...     print(CFG["output_dir"])
    ...     print(CFG["log_level"])
    PosixPath('/path/to/output')
    debug
    >>> print(CFG["output_dir"])
    PosixPath('/home/user/esmvaltool_output')
    >>> print(CFG["log_level"])
    info

By default, the configuration is loaded from YAML files in the user's home
directory at ``~/.config/esmvaltool``.
If set, this can be overwritten with the ``ESMVALTOOL_CONFIG_DIR`` environment
variable.
Defaults for options that are not specified explicitly are listed :ref:`here
<config_options>`.
To reload the current configuration object according to these rules, use:

.. code-block:: python

    >>> CFG.reload()

To load the configuration object from custom directories, use:

.. code-block:: python

    >>> dirs = ['my/default/config', 'my/custom/config']
    >>> CFG.load_from_dirs(dirs)

To update the existing configuration object from custom directories, use:

.. code-block:: python

    >>> dirs = ['my/default/config', 'my/custom/config']
    >>> CFG.update_from_dirs(dirs)


Session
*******

Recipes and diagnostics will be run in their own directories.
This behavior can be controlled via the :py:data:`~esmvalcore.config.Session`
object.
A :py:data:`~esmvalcore.config.Session` must always be initiated from the
global :py:data:`~esmvalcore.config.CFG` object:

.. code-block:: python

    >>> session = CFG.start_session(name='my_session')

A :py:data:`~esmvalcore.config.Session` is very similar to the config.
It is also a dictionary, and copies all the keys from the
:py:data:`~esmvalcore.config.CFG` object.
At this moment, ``session`` is essentially a copy of
:py:data:`~esmvalcore.config.CFG`:

.. code-block:: python

    >>> print(session == CFG)
    True
    >>> session['output_dir'] = '~/my_output_dir'
    >>> print(session == CFG)  # False
    False

A :py:data:`~esmvalcore.config.Session` also knows about the directories where
the data will stored.
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

Unlike the global configuration, of which only one can exist, multiple sessions
can be initiated from :py:data:`~esmvalcore.config.CFG`.


API reference
*************

.. automodule:: esmvalcore.config
    :no-inherited-members:
    :no-show-inheritance:
