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

import contextlib
from pathlib import Path, PosixPath

import iris
import yaml.representer

from esmvalcore.config._config_object import CFG, Config, Session

__all__ = (
    "CFG",
    "Config",
    "Session",
)

# Set iris.FUTURE flags
for attr, value in {
    "save_split_attrs": True,
    "date_microseconds": True,
    "derived_bounds": True,
}.items():
    with contextlib.suppress(AttributeError):
        setattr(iris.FUTURE, attr, value)


# Add a representer for pathlib objects to the YAML library.
def path_representer(
    dumper: yaml.representer.SafeRepresenter,
    data: Path | PosixPath,
) -> yaml.representer.ScalarNode:
    """For printing pathlib.Path objects in yaml files."""
    return dumper.represent_scalar("tag:yaml.org,2002:str", str(data))


yaml.representer.SafeRepresenter.add_representer(Path, path_representer)
yaml.representer.SafeRepresenter.add_representer(PosixPath, path_representer)
