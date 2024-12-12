"""Configuration for Dask distributed."""

import contextlib
import importlib
import logging
import warnings
from collections.abc import Generator, Mapping
from pathlib import Path
from pprint import pformat
from typing import Any

import dask.config
import yaml
from distributed import Client
from distributed.deploy import Cluster

from esmvalcore.config import CFG
from esmvalcore.exceptions import (
    ESMValCoreDeprecationWarning,
    InvalidConfigParameter,
)

logger = logging.getLogger(__name__)

# TODO: Remove in v2.14.0
CONFIG_FILE = Path.home() / ".esmvaltool" / "dask.yml"


# TODO: Remove in v2.14.0
def warn_if_old_dask_config_exists() -> None:
    """Warn user if deprecated dask configuration file exists."""
    if CONFIG_FILE.exists():
        deprecation_msg = (
            "Usage of Dask configuration file ~/.esmvaltool/dask.yml "
            "has been deprecated in ESMValCore version 2.12.0 and is "
            "scheduled for removal in version 2.14.0. Please use the "
            "configuration option `dask` instead. Ignoring all existing "
            "`dask` configuration options for this run."
        )
        warnings.warn(
            deprecation_msg, ESMValCoreDeprecationWarning, stacklevel=2
        )


def validate_dask_config(dask_config: Mapping) -> None:
    """Validate dask configuration options."""
    # Option (1): cluster is specified via address
    if "address" in dask_config.get("client", {}):
        return

    # Option (2): cluster is selected via `run` and `clusters` keys
    for option in ("clusters", "run"):
        if option not in dask_config:
            raise InvalidConfigParameter(
                f"Key '{option}' needs to be defined for 'dask' configuration"
            )
    clusters = dask_config["clusters"]
    run = dask_config["run"]
    if not isinstance(clusters, Mapping):
        raise InvalidConfigParameter(
            f"Key 'dask.clusters' needs to be a mapping, got "
            f"{type(clusters)}"
        )
    for cluster, cluster_config in clusters.items():
        if "type" not in cluster_config:
            raise InvalidConfigParameter(
                f"Key 'dask.clusters.{cluster}' does not have a 'type'"
            )
    if run not in clusters:
        raise InvalidConfigParameter(
            f"Key 'dask.run' needs to point to an element of 'dask.clusters'; "
            f"got '{run}', expected one of {list(clusters.keys())}"
        )


def _process_dask_config_options(dask_config: Mapping) -> None:
    """Process dask options via dask.config.set()."""
    dask_options = dask_config.get("config", {})
    logger.debug(
        "Setting additional Dask options via dask.config.set:\n%s",
        pformat(dask_options),
    )
    dask.config.set(dask_options)


# TODO: Remove in v2.14.0
def _get_old_dask_config() -> dict:
    """Get dask configuration dict from old dask configuration file."""
    dask_config: dict[str, Any] = {"client": {}, "clusters": {}}
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))

    # File exists, but is empty -> Use default scheduler
    if config is None:
        dask_config["client"] = {}
        dask_config["clusters"] = {"default": {"type": "default"}}
        dask_config["run"] = "default"

    # Otherwise, use settings from file
    else:
        client_kwargs = config.get("client", {})
        cluster_kwargs = config.get("cluster", {})

        if "address" in client_kwargs and cluster_kwargs:
            logger.warning(
                "Not using Dask 'cluster' settings from %s because a cluster "
                "'address' is already provided in 'client'.",
                CONFIG_FILE,
            )

        if not cluster_kwargs:
            cluster_kwargs = {"type": "default"}
        if "type" not in cluster_kwargs:
            cluster_kwargs["type"] = "distributed.LocalCluster"

        dask_config["client"] = client_kwargs
        dask_config["clusters"] = {"cluster_from_file": cluster_kwargs}
        dask_config["run"] = "cluster_from_file"

    return dask_config


# TODO: Remove in v2.14.0; used CFG["dask"] instead
def _get_dask_config() -> dict:
    """Get Dask configuration dictionary."""
    if CONFIG_FILE.exists():
        dask_config = _get_old_dask_config()
    else:
        dask_config = CFG["dask"]
    return dask_config


def _setup_cluster(**kwargs) -> None | Cluster:
    """Set up cluster from keyword arguments."""
    kwargs = dict(kwargs)
    cluster_type = kwargs.pop("type", "default")

    # For default cluster, interpret kwargs as keyword arguments for
    # dask.config.set
    if cluster_type == "default":
        logger.debug("Using default Dask cluster with settings %s", kwargs)
        dask.config.set(kwargs)
        return None

    # Otherwise, load cluster class and set up cluster instance
    cluster_module_name, cluster_cls_name = cluster_type.rsplit(".", 1)
    cluster_module = importlib.import_module(cluster_module_name)
    cluster_cls = getattr(cluster_module, cluster_cls_name)
    cluster = cluster_cls(**kwargs)
    logger.debug("Using internal Dask cluster %s", cluster)
    return cluster


@contextlib.contextmanager
def get_distributed_client() -> Generator[None | Client]:
    """Get a Dask distributed client."""
    warn_if_old_dask_config_exists()
    dask_config = _get_dask_config()
    validate_dask_config(dask_config)
    _process_dask_config_options(dask_config)

    client_kwargs = dask_config.get("client", {})
    client: None | Client
    cluster: None | Cluster

    # Client address is defined -> Use that one
    if "address" in client_kwargs:
        logger.info(
            "Using external Dask cluster at %s", client_kwargs["address"]
        )
        client = Client(**client_kwargs)
        cluster = None

    # Otherwise, setup cluster according to the selected entry
    # Note: we already ensured earlier that the selected entry (via `run`)
    # actually exists in `clusters`, so we don't have to check that again here
    else:
        logger.debug("Selecting Dask cluster '%s'", dask_config["run"])
        cluster_kwargs = dask_config["clusters"][dask_config["run"]]
        cluster = _setup_cluster(**cluster_kwargs)
        if cluster is None:
            client = None
        else:
            client_kwargs["address"] = cluster.scheduler_address
            client = Client(**client_kwargs)

    if client:
        logger.info(
            "Using Dask distributed scheduler (dashboard link: %s)",
            client.dashboard_link,
        )
    else:
        logger.warning(
            "Using Dask default scheduler, checkout "
            "https://docs.esmvaltool.org/projects/ESMValCore/en/latest/"
            "quickstart/configure.html#dask-configuration how to use a "
            "distributed scheduler"
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
