"""Tests for ``_get_rootpath`` in ``esmvalcore.local``."""
from unittest import mock

import pytest

from esmvalcore import local


@mock.patch("os.path.exists")
def test_get_rootpath_exists(mexists):
    mexists.return_value = True
    cfg = {"rootpath": {"CMIP5": ["/path1"], "CMIP6": ["/path2"]}}
    project = "CMIP5"
    with mock.patch.dict(local.CFG, cfg):
        output = local._get_rootpath(project)
    # 'output' is a list containing a PosixPath:
    assert str(output[0]) == cfg["rootpath"][project][0]


@mock.patch("os.path.exists")
def test_get_rootpath_does_not_exist(mexists):
    mexists.return_value = False
    cfg = {"rootpath": {"CMIP5": ["path1"], "CMIP6": ["path2"]}}
    project = "OBS"
    with mock.patch.dict(local.CFG, cfg):
        msg = rf"The \"{project}\" option is missing.*"
        with pytest.raises(KeyError, match=msg):
            local._get_rootpath(project)
