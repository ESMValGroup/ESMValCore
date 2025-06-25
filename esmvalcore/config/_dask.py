"""Configuration for Dask distributed."""

import contextlib
import importlib
import logging
import os
import warnings
from collections.abc import Generator, Mapping
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import dask.config
import yaml
from distributed import Client

from esmvalcore.config import CFG
from esmvalcore.exceptions import (
    ESMValCoreDeprecationWarning,
    InvalidConfigParameter,
)

if TYPE_CHECKING:
    from distributed.deploy import Cluster

logger = logging.getLogger(__name__)

# TODO: Remove in v2.14.0
CONFIG_FILE = Path.home() / ".esmvaltool" / "dask.yml"


# TODO: Remove in v2.14.0
def warn_if_old_dask_config_exists() -> None:
    """Warn user if deprecated dask configuration file exists."""
    if CONFIG_FILE.exists() and not os.environ.get(
        "ESMVALTOOL_USE_NEW_DASK_CONFIG",
    ):
        deprecation_msg = (
            "Usage of Dask configuration file ~/.esmvaltool/dask.yml "
            "has been deprecated in ESMValCore version 2.12.0 and is "
            "scheduled for removal in version 2.14.0. Please use the "
            "configuration option `dask` instead (see "
            "https://docs.esmvaltool.org/projects/ESMValCore/en/latest/"
            "quickstart/configure.html#dask-configuration for details). "
            "Ignoring all existing `dask` configuration options for this run. "
            "To enable the new `dask` configuration options, delete or move "
            "the file ~/.esmvaltool/dask.yml or set the environment variable "
            "ESMVALTOOL_USE_NEW_DASK_CONFIG=1."
        )
        warnings.warn(
            deprecation_msg,
            ESMValCoreDeprecationWarning,
            stacklevel=2,
        )


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


# TODO: Remove in v2.14.0
def _get_old_dask_config() -> dict:
    """Get dask configuration dict from old dask configuration file."""
    dask_config: dict[str, Any] = {
        "use": "local_threaded",
        "profiles": {"local_threaded": {"scheduler": "threads"}},
    }
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))

    # Use settings from file if this is not empty
    if config is not None:
        client_kwargs = config.get("client", {})
        cluster_kwargs = config.get("cluster", {})

        # Externally managed cluster
        if "address" in client_kwargs:
            if cluster_kwargs:
                logger.warning(
                    "Not using Dask 'cluster' settings from %s because a "
                    "cluster 'address' is already provided in 'client'.",
                    CONFIG_FILE,
                )
            dask_config = {
                "use": "external",
                "profiles": {
                    "external": {
                        "scheduler_address": client_kwargs.pop("address"),
                    },
                },
            }

        # Dask distributed cluster
        elif cluster_kwargs:
            cluster_kwargs.setdefault("type", "distributed.LocalCluster")
            dask_config = {
                "use": "cluster_from_file",
                "profiles": {
                    "cluster_from_file": {
                        "cluster": cluster_kwargs,
                    },
                },
            }

        dask_config["client"] = client_kwargs

    return dask_config


# TODO: Remove in v2.14.0; used deepcopy(CFG["dask"]) instead
def _get_dask_config() -> dict:
    """Get Dask configuration dictionary."""
    if CONFIG_FILE.exists() and not os.environ.get(
        "ESMVALTOOL_USE_NEW_DASK_CONFIG",
    ):
        dask_config = _get_old_dask_config()
    else:
        dask_config = deepcopy(CFG["dask"])
    return dask_config


@contextlib.contextmanager
def get_distributed_client() -> Generator[None | Client]:
    """Get a Dask distributed client."""
    warn_if_old_dask_config_exists()
    dask_config = _get_dask_config()
    validate_dask_config(dask_config)

    # TODO: Remove in v2.14.0
    client_kwargs = dask_config.get("client", {})

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
        client = Client(**client_kwargs)
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
