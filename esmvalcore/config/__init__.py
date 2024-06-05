"""Configuration module.

.. data:: CFG

    Global ESMValCore configuration object of type
    :class:`esmvalcore.config.Config`.

    By default, this will be loaded from YAML files in
    ``~/.config/esmvaltool``, similar to the way `Dask handles configuration
    <https://docs.dask.org/en/stable/configuration.html>`__. If used within the
    ``esmvaltool`` program, the directory given by ``--config_dir`` will be
    read instead.

    In addition, the environment variable ``ESMVALTOOL_CONFIG_DIR`` is
    considered (with higher priority than the directories mentioned above).

"""

from ._config_object import CFG, Config, Session

__all__ = (
    'CFG',
    'Config',
    'Session',
)
