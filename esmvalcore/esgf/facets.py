"""Find files on the ESGF and download them.

.. deprecated:: 2.14.0
    This module has been moved to :mod:`esmvalcore.io.esgf.facets`. Importing it as
    :mod:`esmvalcore.io.esgf.facets` is deprecated and will be removed in version 2.16.0.
"""

from esmvalcore.io.esgf.facets import DATASET_MAP, FACETS, create_dataset_map

__all__ = [
    "DATASET_MAP",
    "FACETS",
    "create_dataset_map",
]

if __name__ == "__main__":
    # Run this module to create an up to date DATASET_MAP
    print(create_dataset_map())  # noqa: T201
