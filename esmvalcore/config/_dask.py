"""Configuration for Dask distributed."""

import contextlib
import importlib
import logging
from collections.abc import Generator, Mapping
from copy import deepcopy
from typing import TYPE_CHECKING

import dask.config
from distributed import Client

from esmvalcore.config import CFG
from esmvalcore.exceptions import (
    InvalidConfigParameter,
)

if TYPE_CHECKING:
    from distributed.deploy import Cluster

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


@contextlib.contextmanager
def get_distributed_client() -> Generator[None | Client]:
    """Get a Dask distributed client."""
    dask_config = deepcopy(CFG["dask"])
    validate_dask_config(dask_config)

    # Set up cluster and client according to the selected profile
    # Note: we already ensured earlier that the selected profile (via `use`)
    # actually exists in `profiles`, so we don't have to check that again here
    logger.debug("Using Dask profile '%s'", dask_config["use"])
    profile = dask_config["profiles"][dask_config["use"]]
    cluster_kwargs = profile.pop("cluster", None)

    logger.debug("Using additional Dask settings %s", profile)
    dask.config.set(profile)

    cluster: None | Cluster
    client: None | Client

    # Threaded scheduler
    if cluster_kwargs is None:
        cluster = None

    # Distributed scheduler
    else:
        cluster_type = cluster_kwargs.pop("type")
        cluster_module_name, cluster_cls_name = cluster_type.rsplit(".", 1)
        cluster_module = importlib.import_module(cluster_module_name)
        cluster_cls = getattr(cluster_module, cluster_cls_name)
        cluster = cluster_cls(**cluster_kwargs)
        dask.config.set({"scheduler_address": cluster.scheduler_address})
        logger.debug("Using Dask cluster %s", cluster)

    if dask.config.get("scheduler_address", None) is None:
        client = None
        logger.info(
            "Using Dask threaded scheduler. The distributed scheduler is "
            "recommended, please read https://docs.esmvaltool.org/projects/"
            "ESMValCore/en/latest/quickstart/"
            "configure.html#dask-configuration how to use a distributed "
            "scheduler.",
        )
    else:
        client = Client()
        logger.info(
            "Using Dask distributed scheduler (address: %s, dashboard link: "
            "%s)",
            dask.config.get("scheduler_address"),
            client.dashboard_link,
        )

    try:
        yield client
    finally:
        if client is not None:
            client.close()
        if cluster is not None:
            try:
                cluster.close()
            except TimeoutError:
                logger.warning(
                    "Timeout while trying to shut down the cluster at %s, "
                    "you may want to check it was stopped.",
                    cluster.scheduler_address,
                )
