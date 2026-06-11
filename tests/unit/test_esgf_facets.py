"""Test the `esmvalcore.esgf.facets` module."""

# Note that the esmvalcore.esgf module has been moved to esmvalcore.io.esgf
# and support for importing it as esmvalcore.esgf will be removed in v2.16.
# These test can be removed in v2.16 too.

import esmvalcore.esgf.facets


def test_facets():
    assert isinstance(esmvalcore.esgf.facets.FACETS, dict)


def test_dataset_map():
    assert isinstance(esmvalcore.esgf.facets.DATASET_MAP, dict)
