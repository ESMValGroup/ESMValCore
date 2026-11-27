from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import dask.config
import pytest
import xarray as xr
from loguru import logger

from esmvalcore.config import _dask
from esmvalcore.exceptions import (
    InvalidConfigParameter,
)

if TYPE_CHECKING:
    from unittest.mock import Mock

    import pytest_mock

    from esmvalcore.config import Session


@pytest.fixture
def mock_dask_config_set(
    mocker: pytest_mock.MockerFixture,
) -> Mock:
    return mocker.spy(dask.config, "set")


def test_get_no_distributed_client(session: Session) -> None:
    with _dask.get_distributed_client(session) as client:
        assert client is None


def test_get_distributed_client_restores_dask_config(
    mocker: pytest_mock.MockerFixture,
    session: Session,
) -> None:
    session["dask"] = {
        "use": "local_distributed",
        "profiles": {
            "local_distributed": {
                "cluster": {"type": "distributed.LocalCluster"},
                "num_workers": 42,
            },
        },
    }
    mocker.patch.object(_dask, "Client", create_autospec=True)
    mock_cluster_cls = mocker.Mock()
    mock_cluster_cls.return_value.scheduler_address = "tcp://127.0.0.1:4242"
    mocker.patch.object(
        _dask.importlib,
        "import_module",
        create_autospec=True,
        return_value=mocker.Mock(LocalCluster=mock_cluster_cls),
    )
    original = deepcopy(dask.config.config)

    with _dask.get_distributed_client(session):
        assert dask.config.get("scheduler_address") == "tcp://127.0.0.1:4242"
        assert dask.config.get("num_workers") == 42

    assert dask.config.config == original


@pytest.mark.usefixtures("mock_dask_config_set")
def test_get_distributed_client_client_fails(
    mocker: pytest_mock.MockerFixture,
    session: Session,
) -> None:
    session["dask"] = {
        "use": "local_distributed",
        "profiles": {
            "local_distributed": {
                "cluster": {"type": "distributed.LocalCluster"},
            },
        },
    }
    mocker.patch.object(
        _dask,
        "Client",
        create_autospec=True,
        side_effect=OSError,
    )
    mock_cluster_cls = mocker.Mock()
    mocker.patch.object(
        _dask.importlib,
        "import_module",
        create_autospec=True,
        return_value=mocker.Mock(LocalCluster=mock_cluster_cls),
    )

    with (
        pytest.raises(OSError),
        _dask.get_distributed_client(session),
    ):
        pass

    mock_cluster_cls.return_value.close.assert_called_once_with()
    assert dask.config.get("scheduler_address", None) is None


def test_xarray_options_plugin(mocker: pytest_mock.MockerFixture) -> None:
    original = xr.get_options()
    plugin = _dask._XarrayOptionsPlugin({"file_cache_maxsize": 2})
    try:
        plugin.setup(mocker.sentinel.worker)
        assert xr.get_options()["file_cache_maxsize"] == 2
    finally:
        xr.set_options(file_cache_maxsize=original["file_cache_maxsize"])


def test_get_distributed_client_external(
    mocker: pytest_mock.MockerFixture,
    mock_dask_config_set: Mock,
    session: Session,
) -> None:
    session["dask"] = {
        "use": "external",
        "profiles": {
            "external": {
                "scheduler_address": "tcp://127.0.0.1:42021",
            },
        },
        "worker_xarray_options": {"file_cache_maxsize": 2},
    }

    # Create mock distributed.Client
    mock_client = mocker.Mock()
    mocker.patch.object(
        _dask,
        "Client",
        create_autospec=True,
        return_value=mock_client,
    )

    mock_register_plugin = mock_client.register_plugin

    with _dask.get_distributed_client(session) as client:
        assert client is mock_client
    _dask.Client.assert_called_once_with()
    mock_client.close.assert_called_once_with()  # type: ignore[union-attr]
    xarray_plugins = [
        c.args[0]
        for c in mock_register_plugin.call_args_list
        if isinstance(c.args[0], _dask._XarrayOptionsPlugin)
    ]
    assert len(xarray_plugins) == 1
    (plugin,) = xarray_plugins
    assert plugin.options == {"file_cache_maxsize": 2}
    assert (
        mocker.call({"scheduler_address": "tcp://127.0.0.1:42021"})
        in mock_dask_config_set.mock_calls
    )


@pytest.mark.usefixtures("mock_dask_config_set")
def test_get_distributed_client_no_xarray_options(
    mocker: pytest_mock.MockerFixture,
    session: Session,
) -> None:
    session["dask"] = {
        "use": "external",
        "profiles": {
            "external": {
                "scheduler_address": "tcp://127.0.0.1:42021",
            },
        },
        "worker_xarray_options": {},
    }
    mock_client = mocker.Mock()
    mock_register_plugin = mock_client.register_plugin
    mocker.patch.object(
        _dask,
        "Client",
        create_autospec=True,
        return_value=mock_client,
    )

    with _dask.get_distributed_client(session) as client:
        assert client is mock_client
    assert not any(
        isinstance(c.args[0], _dask._XarrayOptionsPlugin)
        for c in mock_register_plugin.call_args_list
    )


@pytest.mark.usefixtures("mock_dask_config_set")
def test_get_distributed_client_register_plugin_fails(
    mocker: pytest_mock.MockerFixture,
    session: Session,
) -> None:
    session["dask"] = {
        "use": "local_distributed",
        "profiles": {
            "local_distributed": {
                "cluster": {"type": "distributed.LocalCluster"},
            },
        },
        "worker_xarray_options": {"file_cache_maxsize": 2},
    }
    mock_client = mocker.Mock()
    mock_client.register_plugin.side_effect = OSError
    mock_close = mock_client.close
    mocker.patch.object(
        _dask,
        "Client",
        create_autospec=True,
        return_value=mock_client,
    )
    mock_cluster_cls = mocker.Mock()
    mocker.patch.object(
        _dask.importlib,
        "import_module",
        create_autospec=True,
        return_value=mocker.Mock(LocalCluster=mock_cluster_cls),
    )

    with (
        pytest.raises(OSError),
        _dask.get_distributed_client(session),
    ):
        pass

    mock_close.assert_called_once_with()
    mock_cluster_cls.return_value.close.assert_called_once_with()


@pytest.mark.parametrize("shutdown_timeout", [False, True])
def test_get_distributed_client_slurm(
    mocker: pytest_mock.MockerFixture,
    mock_dask_config_set: Mock,
    session: Session,
    shutdown_timeout: bool,
) -> None:
    slurm_cluster = {
        "type": "dask_jobqueue.SLURMCluster",
        "queue": "interactive",
        "cores": "8",
        "memory": "16GiB",
    }
    session["dask"] = {
        "use": "slurm_cluster",
        "profiles": {
            "slurm_cluster": {
                "cluster": slurm_cluster,
                "num_workers": 42,
            },
        },
    }

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
    with _dask.get_distributed_client(session) as client:
        assert client is mock_client
    mock_client.close.assert_called_once_with()  # type: ignore[union-attr]
    _dask.Client.assert_called_once_with()
    args = {k: v for k, v in slurm_cluster.items() if k != "type"}
    mock_cluster_cls.assert_called_once_with(**args)
    mock_cluster.close.assert_called()
    assert mocker.call({"num_workers": 42}) in mock_dask_config_set.mock_calls
    assert (
        mocker.call({"scheduler_address": mock_cluster.scheduler_address})
        in mock_dask_config_set.mock_calls
    )


def test_custom_default_scheduler(
    mock_dask_config_set: Mock,
    session: Session,
) -> None:
    default_scheduler = {"num_workers": 42, "scheduler": "processes"}
    session["dask"] = {
        "use": "process_scheduler",
        "profiles": {"process_scheduler": default_scheduler},
        "worker_xarray_options": {"file_cache_maxsize": 2},
    }

    with _dask.get_distributed_client(session) as client:
        assert client is None

    mock_dask_config_set.assert_called_with(
        {"num_workers": 42, "scheduler": "processes"},
    )


def test_invalid_dask_config_no_profiles(session: Session) -> None:
    session["dask"] = {}

    msg = "Key 'profiles' needs to be defined for 'dask' configuration"
    with pytest.raises(InvalidConfigParameter, match=msg):
        with _dask.get_distributed_client(session):
            pass


def test_invalid_dask_config_no_use(session: Session) -> None:
    session["dask"] = {"profiles": {}}

    msg = "Key 'use' needs to be defined for 'dask' configuration"
    with pytest.raises(InvalidConfigParameter, match=msg):
        with _dask.get_distributed_client(session):
            pass


def test_invalid_dask_config_invalid_profiles(session: Session) -> None:
    session["dask"] = {"use": "test", "profiles": 1}

    msg = "Key 'dask.profiles' needs to be a mapping, got"
    with pytest.raises(InvalidConfigParameter, match=msg):
        with _dask.get_distributed_client(session):
            pass


@pytest.mark.parametrize(
    "address_name",
    ["scheduler_address", "scheduler-address"],
)
def test_invalid_dask_config_profile_with_cluster_and_address(
    session: Session,
    address_name: str,
) -> None:
    session["dask"] = {
        "use": "test",
        "profiles": {
            "test": {
                "cluster": {},
                address_name: "8786",
            },
        },
    }

    msg = "Key 'dask.profiles.test' uses 'cluster' and 'scheduler_address'"
    with pytest.raises(InvalidConfigParameter, match=msg):
        with _dask.get_distributed_client(session):
            pass


def test_invalid_dask_config_profile_invalid_cluster(session: Session) -> None:
    session["dask"] = {
        "use": "test",
        "profiles": {
            "test": {"cluster": 1},
        },
    }

    msg = "Key 'dask.profiles.test.cluster' needs to be a mapping"
    with pytest.raises(InvalidConfigParameter, match=msg):
        with _dask.get_distributed_client(session):
            pass


def test_invalid_dask_config_cluster_no_type(session: Session) -> None:
    session["dask"] = {
        "use": "test",
        "profiles": {
            "test": {"cluster": {}},
        },
    }

    msg = "Key 'dask.profiles.test.cluster' does not have a 'type'"
    with pytest.raises(InvalidConfigParameter, match=msg):
        with _dask.get_distributed_client(session):
            pass


def test_invalid_dask_config_invalid_use(session: Session) -> None:
    session["dask"] = {
        "use": "not_in_profiles",
        "profiles": {
            "test": {},
        },
    }

    msg = "Key 'dask.use' needs to point to an element of 'dask.profiles'"
    with pytest.raises(InvalidConfigParameter, match=msg):
        with _dask.get_distributed_client(session):
            pass


@pytest.mark.parametrize(
    ("worker_xarray_options", "msg"),
    [
        (1, "Key 'dask.worker_xarray_options' needs to be a mapping, got"),
        (
            {"not_an_option": 1},
            "Key 'dask.worker_xarray_options' is invalid:",
        ),
        (
            {"file_cache_maxsize": -1},
            "Key 'dask.worker_xarray_options' is invalid:",
        ),
    ],
)
def test_invalid_dask_config_worker_xarray_options(
    session: Session,
    worker_xarray_options: object,
    msg: str,
) -> None:
    session["dask"] = {
        "use": "test",
        "profiles": {
            "test": {},
        },
        "worker_xarray_options": worker_xarray_options,
    }

    with pytest.raises(InvalidConfigParameter, match=msg):
        with _dask.get_distributed_client(session):
            pass


def test_forward_logs_plugin_setup_teardown(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Test that the plugin stops sending messages after teardown."""
    # Keep the default handler of the test process.
    remove = logger.remove
    mocker.patch.object(
        logger,
        "remove",
        side_effect=lambda handler_id: (
            None if handler_id == 0 else remove(handler_id)
        ),
    )
    worker = mocker.Mock()
    plugin = _dask._ForwardLogsPlugin()
    # Pretend that the worker runs in a different process than the client.
    plugin._client_pid = -1
    test_logger = logger.patch(lambda r: r.update(name="esmvalcore.test"))

    plugin.setup(worker)
    test_logger.debug("before teardown")
    plugin.teardown(worker)
    test_logger.debug("after teardown")

    worker.log_event.assert_called_once()
    topic, msg = worker.log_event.call_args.args
    assert topic == _dask._WORKER_LOG_TOPIC
    assert msg["name"] == "esmvalcore.test"
    assert msg["level"] == "DEBUG"
    assert msg["message"] == "before teardown"
    assert plugin._handler_id is None

    # A second teardown does nothing.
    plugin.teardown(worker)
