"""Configuration module.

.. data:: CFG

    Global ESMValCore configuration object of type
    :class:`esmvalcore.config.Config`.

    By default, this will be loaded from YAML files in the user configuration
    directory (by default ``~/.config/esmvaltool``, but this can be changed
    with the ``ESMVALTOOL_CONFIG_DIR`` environment variable) similar to the way
    `Dask handles configuration
    <https://docs.dask.org/en/stable/configuration.html>`__.

"""

from ._config_object import CFG, Config, Session

__all__ = (
    "CFG",
    "Config",
    "Session",
)
