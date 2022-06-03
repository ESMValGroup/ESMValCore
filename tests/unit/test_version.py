"""Test that esmvalcore.__version__ returns a version number."""
import re

import esmvalcore


def test_version():

    assert re.match(r"^\d+\.\d+\.\d+\S*$", esmvalcore.__version__)
