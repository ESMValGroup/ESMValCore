"""Configuration for Dask distributed."""

from __future__ import annotations

import contextlib
import importlib
import os
from collections.abc import Mapping
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import dask.config
import xarray as xr
from distributed import Client, WorkerPlugin
from loguru import logger

from esmvalcore.config._logging import _debug_log_filter
from esmvalcore.exceptions import InvalidConfigParameter

if TYPE_CHECKING:
    from collections.abc import Generator

    from distributed import Worker
    from distributed.deploy import Cluster
    from loguru import Message

    from esmvalcore.config import Session


def validate_dask_config(dask_config: Mapping) -> None:
    """Validate dask configuration options."""
    for option in ("profiles", "use"):
        if option not in dask_config:
            msg = (
                f"Key '{option}' needs to be defined for 'dask' configuration"
            )
            raise InvalidConfigParameter(msg)
    profiles = dask_config["profiles"]
    use = dask_config["use"]
    if not isinstance(profiles, Mapping):
        msg = (
            f"Key 'dask.profiles' needs to be a mapping, got {type(profiles)}"
        )
        raise InvalidConfigParameter(msg)
    for profile, profile_cfg in profiles.items():
        has_scheduler_address = any(
            [
                "scheduler_address" in profile_cfg,
                "scheduler-address" in profile_cfg,
            ],
        )
        if "cluster" in profile_cfg and has_scheduler_address:
            msg = (
                f"Key 'dask.profiles.{profile}' uses 'cluster' and "
                f"'scheduler_address', can only have one of those"
            )
            raise InvalidConfigParameter(msg)
        if "cluster" in profile_cfg:
            cluster = profile_cfg["cluster"]
            if not isinstance(cluster, Mapping):
                msg = (
                    f"Key 'dask.profiles.{profile}.cluster' needs to be a "
                    f"mapping, got {type(cluster)}"
                )
                raise InvalidConfigParameter(msg)
            if "type" not in cluster:
                msg = (
                    f"Key 'dask.profiles.{profile}.cluster' does not have a "
                    f"'type'"
                )
                raise InvalidConfigParameter(msg)
    if use not in profiles:
        msg = (
            f"Key 'dask.use' needs to point to an element of 'dask.profiles'; "
            f"got '{use}', expected one of {list(profiles.keys())}"
        )
        raise InvalidConfigParameter(msg)
    _validate_worker_xarray_options(
        dask_config.get("worker_xarray_options", {}),
    )


def _validate_worker_xarray_options(xarray_options: object) -> None:
    """Validate the xarray options for the Dask workers."""
    if not isinstance(xarray_options, Mapping):
        msg = (
            f"Key 'dask.worker_xarray_options' needs to be a mapping, got "
            f"{type(xarray_options)}"
        )
        raise InvalidConfigParameter(msg)
    try:
        with xr.set_options(**xarray_options):
            pass
    except (TypeError, ValueError) as exc:
        msg = f"Key 'dask.worker_xarray_options' is invalid: {exc}"
        raise InvalidConfigParameter(msg) from exc


class _XarrayOptionsPlugin(WorkerPlugin):
    """Set :func:`xarray.set_options` on each Dask worker."""

    name = "esmvalcore-xarray-options"

    def __init__(self, options: Mapping[str, Any]) -> None:
        self.options = dict(options)

    def setup(self, worker: Worker) -> None:  # noqa: ARG002
        """Set the xarray options when the worker starts."""
        xr.set_options(**self.options)


_WORKER_LOG_TOPIC = "esmvalcore-log"
"""Dask event topic used to send log messages from the workers to the client."""


def _send_to_client(worker: Worker, message: Message) -> None:
    """Send a log message from a Dask worker to the client."""
    record = message.record
    worker.log_event(
        _WORKER_LOG_TOPIC,
        {
            "name": record["name"],
            "level": record["level"].name,
            "line": record["line"],
            "function": record["function"],
            # The message includes the traceback if there is an exception.
            "message": str(message).rstrip("\n"),
        },
    )


def _log_worker_message(event: tuple[float, dict[str, Any]]) -> None:
    """Log a message received from a Dask worker."""
    _, msg = event
    logger.patch(
        lambda r: r.update(
            name=msg["name"],
            line=msg["line"],
            function=msg["function"],
        ),
    ).log(msg["level"], msg["message"])


class _ForwardLogsPlugin(WorkerPlugin):
    """Send log messages from Dask workers to the client.

    This makes sure that messages logged by ESMValCore code running on the
    workers end up in the log files, which are written by the client process
    only.
    """

    def __init__(self) -> None:
        self._client_pid = os.getpid()
        self._handler_id: int | None = None

    def setup(self, worker: Worker) -> None:
        if os.getpid() == self._client_pid:
            # The worker runs in the client process, e.g. a LocalCluster
            # with processes=False, so its messages are already logged there.
            return
        # Replace the default handler, which prints all messages to stderr.
        with contextlib.suppress(ValueError):
            logger.remove(0)
        logger.enable("esmvalcore")
        self._handler_id = logger.add(
            lambda message: _send_to_client(worker, message),
            level="DEBUG",
            format="{message}",
            filter=_debug_log_filter,
            backtrace=False,
            diagnose=False,
        )

    def teardown(self, worker: Worker) -> None:  # noqa: ARG002
        if self._handler_id is not None:
            logger.remove(self._handler_id)
            self._handler_id = None


def _forward_worker_logs(client: Client) -> None:
    """Log messages from the Dask workers in the client process."""
    client.subscribe_topic(_WORKER_LOG_TOPIC, _log_worker_message)
    client.register_plugin(_ForwardLogsPlugin(), name="esmvalcore-logging")


@contextlib.contextmanager
def get_distributed_client(session: Session) -> Generator[Client | None]:
    """Get a Dask distributed client."""
    dask_config = deepcopy(session["dask"])
    validate_dask_config(dask_config)
    xarray_options = dask_config.pop("worker_xarray_options", {})

    # Set up cluster and client according to the selected profile
    # Note: we already ensured earlier that the selected profile (via `use`)
    # actually exists in `profiles`, so we don't have to check that again here
    logger.debug("Using Dask profile '{}'", dask_config["use"])
    profile = dask_config["profiles"][dask_config["use"]]
    cluster_kwargs = profile.pop("cluster", None)

    logger.debug("Using additional Dask settings {}", profile)

    # Use an exit stack, so the Dask configuration is restored and the
    # client and cluster are closed when leaving the context, even if an
    # error occurs while setting them up.
    with contextlib.ExitStack() as stack:
        stack.enter_context(dask.config.set(profile))

        # Threaded scheduler: nothing to set up.
        # Distributed scheduler: start the cluster.
        if cluster_kwargs is not None:
            cluster_type = cluster_kwargs.pop("type")
            cluster_module_name, cluster_cls_name = cluster_type.rsplit(".", 1)
            cluster_module = importlib.import_module(cluster_module_name)
            cluster_cls = getattr(cluster_module, cluster_cls_name)
            cluster = cluster_cls(**cluster_kwargs)
            stack.callback(_close_cluster, cluster)
            stack.enter_context(
                dask.config.set(
                    {"scheduler_address": cluster.scheduler_address},
                ),
            )
            logger.debug("Using Dask cluster {}", cluster)

        client: Client | None
        if dask.config.get("scheduler_address", None) is None:
            client = None
            logger.info(
                "Using Dask threaded scheduler. The distributed scheduler is "
                "recommended, please read https://docs.esmvaltool.org/"
                "projects/ESMValCore/en/latest/quickstart/"
                "configure.html#dask-configuration how to use a distributed "
                "scheduler.",
            )
        else:
            client = Client()
            stack.callback(client.close)
            _forward_worker_logs(client)
            logger.info(
                "Using Dask distributed scheduler (address: {}, dashboard "
                "link: {})",
                dask.config.get("scheduler_address"),
                client.dashboard_link,
            )
            if xarray_options:
                client.register_plugin(_XarrayOptionsPlugin(xarray_options))

        yield client


def _close_cluster(cluster: Cluster) -> None:
    """Close a Dask cluster."""
    try:
        cluster.close()
    except TimeoutError:
        logger.warning(
            "Timeout while trying to shut down the cluster at {}, "
            "you may want to check it was stopped.",
            cluster.scheduler_address,
        )
