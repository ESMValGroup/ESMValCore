import pytest
import yaml

from esmvalcore.config import CFG, _dask
from esmvalcore.exceptions import (
    ESMValCoreDeprecationWarning,
    InvalidConfigParameter,
)


@pytest.fixture
def mock_dask_config_set(mocker):
    dask_config_dict = {}
    mock_dask_set = mocker.patch("dask.config.set", autospec=True)
    mock_dask_set.side_effect = dask_config_dict.update
    mock_dask_get = mocker.patch("dask.config.get", autospec=True)
    mock_dask_get.side_effect = dask_config_dict.get
    return mock_dask_set


def test_get_no_distributed_client(ignore_existing_user_config):
    with _dask.get_distributed_client() as client:
        assert client is None


# TODO: Remove in v2.14.0
def test_get_distributed_client_empty_dask_file(mocker, tmp_path):
    # Create mock client configuration.
    cfg_file = tmp_path / "dask.yml"
    with cfg_file.open("w", encoding="utf-8") as file:
        file.write("")
    mocker.patch.object(_dask, "CONFIG_FILE", cfg_file)

    # Create mock distributed.Client
    with pytest.warns(ESMValCoreDeprecationWarning):
        with _dask.get_distributed_client() as client:
            assert client is None


# TODO: Remove in v2.14.0
@pytest.mark.parametrize("use_new_dask_config", ["", "1"])
def test_force_new_dask_config(
    monkeypatch,
    mocker,
    tmp_path,
    mock_dask_config_set,
    ignore_existing_user_config,
    use_new_dask_config,
):
    # Old config -> threaded scheduler
    cfg_file = tmp_path / "dask.yml"
    with cfg_file.open("w", encoding="utf-8") as file:
        file.write("")
    mocker.patch.object(_dask, "CONFIG_FILE", cfg_file)

    # New config -> distributed scheduler
    slurm_cluster = {
        "type": "dask_jobqueue.SLURMCluster",
        "queue": "interactive",
        "cores": "8",
        "memory": "16GiB",
    }
    monkeypatch.setitem(
        CFG,
        "dask",
        {
            "use": "slurm_cluster",
            "profiles": {"slurm_cluster": {"cluster": slurm_cluster}},
        },
    )

    # Create mock distributed.Client
    mock_client = mocker.Mock()
    mocker.patch.object(
        _dask,
        "Client",
        create_autospec=True,
        return_value=mock_client,
    )

    mock_module = mocker.Mock()
    mock_cluster_cls = mocker.Mock()
    mock_module.SLURMCluster = mock_cluster_cls
    mocker.patch.object(
        _dask.importlib,
        "import_module",
        create_autospec=True,
        return_value=mock_module,
    )

    monkeypatch.setenv("ESMVALTOOL_USE_NEW_DASK_CONFIG", use_new_dask_config)

    with _dask.get_distributed_client() as client:
        if use_new_dask_config:
            assert client is not None
        else:
            assert client is None


# TODO: Remove in v2.14.0
def test_get_old_dask_config(mocker, tmp_path):
    # Create mock client configuration.
    cfg = {"cluster": {"n_workers": 2}}
    cfg_file = tmp_path / "dask.yml"
    with cfg_file.open("w", encoding="utf-8") as file:
        yaml.safe_dump(cfg, file)
    mocker.patch.object(_dask, "CONFIG_FILE", cfg_file)

    dask_cfg = _dask._get_old_dask_config()

    expected_cfg = {
        "use": "cluster_from_file",
        "profiles": {
            "cluster_from_file": {
                "cluster": {
                    "n_workers": 2,
                    "type": "distributed.LocalCluster",
                },
            },
        },
        "client": {},
    }
    assert dask_cfg == expected_cfg


def test_get_distributed_client_external(
    monkeypatch,
    mocker,
    mock_dask_config_set,
    ignore_existing_user_config,
):
    monkeypatch.setitem(
        CFG,
        "dask",
        {
            "use": "external",
            "profiles": {
                "external": {
                    "scheduler_address": "tcp://127.0.0.1:42021",
                },
            },
        },
    )

    # Create mock distributed.Client
    mock_client = mocker.Mock()
    mocker.patch.object(
        _dask,
        "Client",
        create_autospec=True,
        return_value=mock_client,
    )

    with _dask.get_distributed_client() as client:
        assert client is mock_client
    _dask.Client.assert_called_once_with()
    mock_client.close.assert_called_once_with()
    assert (
        mocker.call({"scheduler_address": "tcp://127.0.0.1:42021"})
        in mock_dask_config_set.mock_calls
    )


# TODO: Remove in v2.14.0
@pytest.mark.parametrize("warn_unused_args", [False, True])
def test_get_distributed_client_external_old(
    mocker,
    tmp_path,
    mock_dask_config_set,
    warn_unused_args,
):
    # Create mock client configuration.
    cfg = {
        "client": {
            "address": "tcp://127.0.0.1:42021",
            "other_client_options": 1,
        },
    }
    if warn_unused_args:
        cfg["cluster"] = {"n_workers": 2}
    cfg_file = tmp_path / "dask.yml"
    with cfg_file.open("w", encoding="utf-8") as file:
        yaml.safe_dump(cfg, file)
    mocker.patch.object(_dask, "CONFIG_FILE", cfg_file)

    # Create mock distributed.Client
    mock_client = mocker.Mock()
    mocker.patch.object(
        _dask,
        "Client",
        create_autospec=True,
        return_value=mock_client,
    )

    with pytest.warns(ESMValCoreDeprecationWarning):
        with _dask.get_distributed_client() as client:
            assert client is mock_client
    _dask.Client.assert_called_once_with(other_client_options=1)
    mock_client.close.assert_called_once_with()
    assert (
        mocker.call({"scheduler_address": "tcp://127.0.0.1:42021"})
        in mock_dask_config_set.mock_calls
    )


@pytest.mark.parametrize("shutdown_timeout", [False, True])
def test_get_distributed_client_slurm(
    monkeypatch,
    mocker,
    mock_dask_config_set,
    ignore_existing_user_config,
    shutdown_timeout,
):
    slurm_cluster = {
        "type": "dask_jobqueue.SLURMCluster",
        "queue": "interactive",
        "cores": "8",
        "memory": "16GiB",
    }
    monkeypatch.setitem(
        CFG,
        "dask",
        {
            "use": "slurm_cluster",
            "profiles": {
                "slurm_cluster": {
                    "cluster": slurm_cluster,
                    "num_workers": 42,
                },
            },
        },
    )

    # Create mock distributed.Client
    mock_client = mocker.Mock()
    mocker.patch.object(
        _dask,
        "Client",
        create_autospec=True,
        return_value=mock_client,
    )

    mock_module = mocker.Mock()
    mock_cluster_cls = mocker.Mock()
    mock_module.SLURMCluster = mock_cluster_cls
    mocker.patch.object(
        _dask.importlib,
        "import_module",
        create_autospec=True,
        return_value=mock_module,
    )
    mock_cluster = mock_cluster_cls.return_value
    if shutdown_timeout:
        mock_cluster.close.side_effect = TimeoutError
    with _dask.get_distributed_client() as client:
        assert client is mock_client
    mock_client.close.assert_called_once_with()
    _dask.Client.assert_called_once_with()
    args = {k: v for k, v in slurm_cluster.items() if k != "type"}
    mock_cluster_cls.assert_called_once_with(**args)
    mock_cluster.close.assert_called()
    assert mocker.call({"num_workers": 42}) in mock_dask_config_set.mock_calls
    assert (
        mocker.call({"scheduler_address": mock_cluster.scheduler_address})
        in mock_dask_config_set.mock_calls
    )


# TODO: Remove in v2.14.0
@pytest.mark.parametrize("shutdown_timeout", [False, True])
def test_get_distributed_client_slurm_old(
    mocker,
    tmp_path,
    mock_dask_config_set,
    shutdown_timeout,
):
    cfg = {
        "cluster": {
            "type": "dask_jobqueue.SLURMCluster",
            "queue": "interactive",
            "cores": "8",
            "memory": "16GiB",
        },
    }
    cfg_file = tmp_path / "dask.yml"
    with cfg_file.open("w", encoding="utf-8") as file:
        yaml.safe_dump(cfg, file)
    mocker.patch.object(_dask, "CONFIG_FILE", cfg_file)

    # Create mock distributed.Client
    mock_client = mocker.Mock()
    mocker.patch.object(
        _dask,
        "Client",
        create_autospec=True,
        return_value=mock_client,
    )

    mock_module = mocker.Mock()
    mock_cluster_cls = mocker.Mock()
    mock_module.SLURMCluster = mock_cluster_cls
    mocker.patch.object(
        _dask.importlib,
        "import_module",
        create_autospec=True,
        return_value=mock_module,
    )
    mock_cluster = mock_cluster_cls.return_value
    if shutdown_timeout:
        mock_cluster.close.side_effect = TimeoutError
    with pytest.warns(ESMValCoreDeprecationWarning):
        with _dask.get_distributed_client() as client:
            assert client is mock_client
    mock_client.close.assert_called_once_with()
    _dask.Client.assert_called_once_with()
    args = {k: v for k, v in cfg["cluster"].items() if k != "type"}
    mock_cluster_cls.assert_called_once_with(**args)
    mock_cluster.close.assert_called()
    assert (
        mocker.call({"scheduler_address": mock_cluster.scheduler_address})
        in mock_dask_config_set.mock_calls
    )


def test_custom_default_scheduler(
    monkeypatch,
    mock_dask_config_set,
    ignore_existing_user_config,
):
    default_scheduler = {"num_workers": 42, "scheduler": "processes"}
    monkeypatch.setitem(
        CFG,
        "dask",
        {
            "use": "process_scheduler",
            "profiles": {"process_scheduler": default_scheduler},
        },
    )

    with _dask.get_distributed_client() as client:
        assert client is None

    mock_dask_config_set.assert_called_with(
        {"num_workers": 42, "scheduler": "processes"},
    )


def test_invalid_dask_config_no_profiles(monkeypatch):
    monkeypatch.setitem(CFG, "dask", {})

    msg = "Key 'profiles' needs to be defined for 'dask' configuration"
    with pytest.raises(InvalidConfigParameter, match=msg):
        with _dask.get_distributed_client():
            pass


def test_invalid_dask_config_no_use(monkeypatch):
    monkeypatch.setitem(CFG, "dask", {"profiles": {}})

    msg = "Key 'use' needs to be defined for 'dask' configuration"
    with pytest.raises(InvalidConfigParameter, match=msg):
        with _dask.get_distributed_client():
            pass


def test_invalid_dask_config_invalid_profiles(monkeypatch):
    monkeypatch.setitem(CFG, "dask", {"use": "test", "profiles": 1})

    msg = "Key 'dask.profiles' needs to be a mapping, got"
    with pytest.raises(InvalidConfigParameter, match=msg):
        with _dask.get_distributed_client():
            pass


@pytest.mark.parametrize(
    "address_name",
    ["scheduler_address", "scheduler-address"],
)
def test_invalid_dask_config_profile_with_cluster_and_address(
    monkeypatch,
    address_name,
):
    monkeypatch.setitem(
        CFG,
        "dask",
        {
            "use": "test",
            "profiles": {
                "test": {"cluster": {}, address_name: "8786"},
            },
        },
    )

    msg = "Key 'dask.profiles.test' uses 'cluster' and 'scheduler_address'"
    with pytest.raises(InvalidConfigParameter, match=msg):
        with _dask.get_distributed_client():
            pass


def test_invalid_dask_config_profile_invalid_cluster(monkeypatch):
    monkeypatch.setitem(
        CFG,
        "dask",
        {
            "use": "test",
            "profiles": {
                "test": {"cluster": 1},
            },
        },
    )

    msg = "Key 'dask.profiles.test.cluster' needs to be a mapping"
    with pytest.raises(InvalidConfigParameter, match=msg):
        with _dask.get_distributed_client():
            pass


def test_invalid_dask_config_cluster_no_type(monkeypatch):
    monkeypatch.setitem(
        CFG,
        "dask",
        {
            "use": "test",
            "profiles": {
                "test": {"cluster": {}},
            },
        },
    )

    msg = "Key 'dask.profiles.test.cluster' does not have a 'type'"
    with pytest.raises(InvalidConfigParameter, match=msg):
        with _dask.get_distributed_client():
            pass


def test_invalid_dask_config_invalid_use(monkeypatch):
    monkeypatch.setitem(
        CFG,
        "dask",
        {
            "use": "not_in_profiles",
            "profiles": {
                "test": {},
            },
        },
    )

    msg = "Key 'dask.use' needs to point to an element of 'dask.profiles'"
    with pytest.raises(InvalidConfigParameter, match=msg):
        with _dask.get_distributed_client():
            pass
