"""Configuration for Dask distributed."""

import contextlib
import importlib
import logging

import yaml
from distributed import Client

from esmvalcore.config import CFG

logger = logging.getLogger(__name__)


def check_distributed_config():
    """Check the Dask distributed configuration."""
    if not CFG["dask_config"].exists():
        logger.warning(
            "Using the Dask basic scheduler. This may lead to slow "
            "computations and out-of-memory errors. "
            "Note that the basic scheduler may still be the best choice for "
            "preprocessor functions that are not lazy. "
            "In that case, you can safely ignore this warning. "
            "See https://docs.esmvaltool.org/projects/ESMValCore/en/latest/"
            "quickstart/configure.html#dask-distributed-configuration for "
            "more information. "
        )


@contextlib.contextmanager
def get_distributed_client():
    """Get a Dask distributed client."""
    dask_args = {}
    if CFG["dask_config"].exists():
        config = yaml.safe_load(CFG["dask_config"].read_text(encoding="utf-8"))
        if config is not None:
            dask_args = config

    client_args = dask_args.get("client") or {}
    cluster_args = dask_args.get("cluster") or {}

    # Start a cluster, if requested
    if "address" in client_args:
        # Use an externally managed cluster.
        cluster = None
        if cluster_args:
            logger.warning(
                "Not using Dask 'cluster' settings from %s because a cluster "
                "'address' is already provided in 'client'.",
                CFG["dask_config"],
            )
    elif cluster_args:
        # Start cluster.
        cluster_type = cluster_args.pop(
            "type",
            "distributed.LocalCluster",
        )
        cluster_module_name, cluster_cls_name = cluster_type.rsplit(".", 1)
        cluster_module = importlib.import_module(cluster_module_name)
        cluster_cls = getattr(cluster_module, cluster_cls_name)
        cluster = cluster_cls(**cluster_args)
        client_args["address"] = cluster.scheduler_address
    else:
        # No cluster configured, use Dask basic scheduler, or a LocalCluster
        # managed through Client.
        cluster = None

    # Start a client, if requested
    if dask_args:
        client = Client(**client_args)
        logger.info("Dask dashboard: %s", client.dashboard_link)
    else:
        logger.info("Using the Dask basic scheduler.")
        client = None

    try:
        yield client
    finally:
        if client is not None:
            client.close()
        if cluster is not None:
            cluster.close()
