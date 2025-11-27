"""Find files on the ESGF and download them.

.. deprecated:: 2.14.0
    This module has been moved to :mod:`esmvalcore.io.esgf`. Importing it as
    :mod:`esmvalcore.io.esgf` is deprecated and will be removed in version 2.16.0.
"""

from esmvalcore.io.esgf import (
    ESGFDataSource,
    ESGFFile,
    download,
    find_files,
)

__all__ = [
    "ESGFDataSource",
    "ESGFFile",
    "download",
    "find_files",
]
