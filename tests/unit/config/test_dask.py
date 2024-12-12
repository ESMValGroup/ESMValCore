import pytest
import yaml
from distributed import Client

from esmvalcore.config import CFG, _dask
from esmvalcore.exceptions import (
    ESMValCoreDeprecationWarning,
    InvalidConfigParameter,
)


def test_get_no_distributed_client():
    with _dask.get_distributed_client() as client:
        assert client is None


# TODO: Remove in v2.14.0
def test_get_distributed_client_empty_dask_file(mocker, tmp_path):
    # Create mock client configuration.
    cfg = {}
    cfg_file = tmp_path / "dask.yml"
    with cfg_file.open("w", encoding="utf-8") as file:
        yaml.safe_dump(cfg, file)
    mocker.patch.object(_dask, "CONFIG_FILE", cfg_file)

    # Create mock distributed.Client
    with pytest.warns(ESMValCoreDeprecationWarning):
        with _dask.get_distributed_client() as client:
            assert client is None


# TODO: Remove in v2.14.0
def test_get_distributed_client_local_cluster_old(mocker, tmp_path):
    # Create mock client configuration.
    cfg = {"cluster": {"n_workers": 2}}
    cfg_file = tmp_path / "dask.yml"
    with cfg_file.open("w", encoding="utf-8") as file:
        yaml.safe_dump(cfg, file)
    mocker.patch.object(_dask, "CONFIG_FILE", cfg_file)

    # Create mock distributed.Client
    with pytest.warns(ESMValCoreDeprecationWarning):
        with _dask.get_distributed_client() as client:
            assert isinstance(client, Client)


def test_get_distributed_client_external(monkeypatch, mocker):
    client_kwargs = {"address": "tcp://127.0.0.1:42021"}
    monkeypatch.setitem(CFG, "dask", {"client": client_kwargs})

    # Create mock distributed.Client
    mock_client = mocker.Mock()
    mocker.patch.object(
        _dask, "Client", create_autospec=True, return_value=mock_client
    )

    with _dask.get_distributed_client() as client:
        assert client is mock_client
    _dask.Client.assert_called_with(**client_kwargs)
    mock_client.close.assert_called()


# TODO: Remove in v2.14.0
@pytest.mark.parametrize("warn_unused_args", [False, True])
def test_get_distributed_client_external_old(
    mocker,
    tmp_path,
    warn_unused_args,
):
    # Create mock client configuration.
    cfg = {
        "client": {
            "address": "tcp://127.0.0.1:42021",
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
        _dask, "Client", create_autospec=True, return_value=mock_client
    )

    with pytest.warns(ESMValCoreDeprecationWarning):
        with _dask.get_distributed_client() as client:
            assert client is mock_client
    _dask.Client.assert_called_with(**cfg["client"])
    mock_client.close.assert_called()


@pytest.mark.parametrize("shutdown_timeout", [False, True])
def test_get_distributed_client_slurm(monkeypatch, mocker, shutdown_timeout):
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
            "run": "slurm_cluster",
            "clusters": {"slurm_cluster": slurm_cluster},
        },
    )

    # Create mock distributed.Client
    mock_client = mocker.Mock()
    mocker.patch.object(
        _dask, "Client", create_autospec=True, return_value=mock_client
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
    mock_client.close.assert_called()
    _dask.Client.assert_called_with(address=mock_cluster.scheduler_address)
    args = {k: v for k, v in slurm_cluster.items() if k != "type"}
    mock_cluster_cls.assert_called_with(**args)
    mock_cluster.close.assert_called()


def test_get_distributed_client_local(monkeypatch, mocker):
    local_cluster = {
        "type": "distributed.LocalCluster",
        "n_workers": 2,
        "threads_per_worker": 2,
        "memory_limit": "4GiB",
    }
    monkeypatch.setitem(
        CFG,
        "dask",
        {
            "run": "local",
            "clusters": {"local": local_cluster},
        },
    )

    with _dask.get_distributed_client() as client:
        assert isinstance(client, Client)


# TODO: Remove in v2.14.0
@pytest.mark.parametrize("shutdown_timeout", [False, True])
def test_get_distributed_client_slurm_old(mocker, tmp_path, shutdown_timeout):
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
        _dask, "Client", create_autospec=True, return_value=mock_client
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
    mock_client.close.assert_called()
    _dask.Client.assert_called_with(address=mock_cluster.scheduler_address)
    args = {k: v for k, v in cfg["cluster"].items() if k != "type"}
    mock_cluster_cls.assert_called_with(**args)
    mock_cluster.close.assert_called()


def test_custom_default_scheduler(monkeypatch, mocker):
    default_cluster = {
        "type": "default",
        "num_workers": 42,
        "scheduler": "processes",
    }
    monkeypatch.setitem(
        CFG,
        "dask",
        {
            "run": "default",
            "clusters": {"default": default_cluster},
        },
    )
    mock_dask_set = mocker.patch("dask.config.set", autospec=True)

    with _dask.get_distributed_client() as client:
        assert client is None

    mock_calls = [
        mocker.call({}),
        mocker.call({"num_workers": 42, "scheduler": "processes"}),
    ]
    assert mock_dask_set.mock_calls == mock_calls


def test_custom_dask_config(monkeypatch, mocker):
    default_cluster = {
        "type": "default",
        "num_workers": 42,
    }
    monkeypatch.setitem(
        CFG,
        "dask",
        {
            "run": "default",
            "config": {"num_workers": 41, "scheduler": "processes"},
            "clusters": {"default": default_cluster},
        },
    )
    mock_dask_set = mocker.patch("dask.config.set", autospec=True)

    with _dask.get_distributed_client() as client:
        assert client is None

    mock_calls = [
        mocker.call({"num_workers": 41, "scheduler": "processes"}),
        mocker.call({"num_workers": 42}),
    ]
    assert mock_dask_set.mock_calls == mock_calls


def test_invalid_dask_config_no_clusters(monkeypatch, mocker):
    monkeypatch.setitem(CFG, "dask", {})

    msg = "Key 'clusters' needs to be defined for 'dask' configuration"
    with pytest.raises(InvalidConfigParameter, match=msg):
        with _dask.get_distributed_client():
            pass


def test_invalid_dask_config_no_run(monkeypatch, mocker):
    monkeypatch.setitem(CFG, "dask", {"clusters": {}})

    msg = "Key 'run' needs to be defined for 'dask' configuration"
    with pytest.raises(InvalidConfigParameter, match=msg):
        with _dask.get_distributed_client():
            pass


def test_invalid_dask_config_invalid_clusters(monkeypatch, mocker):
    monkeypatch.setitem(
        CFG,
        "dask",
        {
            "run": "test",
            "clusters": {
                "test": {},
            },
        },
    )

    msg = "Key 'dask.clusters.test' does not have a 'type'"
    with pytest.raises(InvalidConfigParameter, match=msg):
        with _dask.get_distributed_client():
            pass


def test_invalid_dask_config_invalid_run(monkeypatch, mocker):
    monkeypatch.setitem(
        CFG,
        "dask",
        {
            "run": "not_in_clusters",
            "clusters": {
                "test": {"type": "default"},
            },
        },
    )

    msg = "Key 'dask.run' needs to point to an element of 'dask.clusters'"
    with pytest.raises(InvalidConfigParameter, match=msg):
        with _dask.get_distributed_client():
            pass
