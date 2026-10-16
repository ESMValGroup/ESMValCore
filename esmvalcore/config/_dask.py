"""Configuration for Dask distributed."""

from __future__ import annotations

import contextlib
import importlib
import logging
from collections.abc import Mapping
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import dask.config
import xarray as xr
from distributed import Client, WorkerPlugin

from esmvalcore.exceptions import InvalidConfigParameter

if TYPE_CHECKING:
    from collections.abc import Generator

    from distributed import Worker
    from distributed.deploy import Cluster

    from esmvalcore.config import Session

logger = logging.getLogger(__name__)


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


@contextlib.contextmanager
def get_distributed_client(session: Session) -> Generator[None | Client]:
    """Get a Dask distributed client."""
    dask_config = deepcopy(session["dask"])
    validate_dask_config(dask_config)
    xarray_options = dask_config.pop("worker_xarray_options", {})

    # Set up cluster and client according to the selected profile
    # Note: we already ensured earlier that the selected profile (via `use`)
    # actually exists in `profiles`, so we don't have to check that again here
    logger.debug("Using Dask profile '%s'", dask_config["use"])
    profile = dask_config["profiles"][dask_config["use"]]
    cluster_kwargs = profile.pop("cluster", None)

    logger.debug("Using additional Dask settings %s", profile)

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
            logger.debug("Using Dask cluster %s", cluster)

        client: None | Client
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
            logger.info(
                "Using Dask distributed scheduler (address: %s, dashboard "
                "link: %s)",
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
            "Timeout while trying to shut down the cluster at %s, "
            "you may want to check it was stopped.",
            cluster.scheduler_address,
        )
