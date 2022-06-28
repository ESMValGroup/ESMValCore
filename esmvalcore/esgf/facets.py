"""Module containing mappings from our names to ESGF names."""

import pyesgf.search

from .._config._esgf_pyclient import get_esgf_config

FACETS = {
    'CMIP3': {
        'dataset': 'model',
        'ensemble': 'ensemble',
        'exp': 'experiment',
        'frequency': 'time_frequency',
        'short_name': 'variable',
    },
    'CMIP5': {
        'dataset': 'model',
        'ensemble': 'ensemble',
        'exp': 'experiment',
        'mip': 'cmor_table',
        'product': 'product',
        'short_name': 'variable',
    },
    'CMIP6': {
        'dataset': 'source_id',
        'ensemble': 'variant_label',
        'exp': 'experiment_id',
        'grid': 'grid_label',
        'mip': 'table_id',
        'short_name': 'variable',
    },
    'CORDEX': {
        'dataset': 'rcm_name',
        'driver': 'driving_model',
        'domain': 'domain',
        'ensemble': 'ensemble',
        'exp': 'experiment',
        'frequency': 'time_frequency',
        'short_name': 'variable',
    },
    'obs4MIPs': {
        'dataset': 'source_id',
        'frequency': 'time_frequency',
        'short_name': 'variable',
    }
}
"""Mapping between the recipe and ESGF facet names."""

DATASET_MAP = {
    'CMIP3': {},
    'CMIP5': {
        'ACCESS1-0': 'ACCESS1.0',
        'ACCESS1-3': 'ACCESS1.3',
        'bcc-csm1-1': 'BCC-CSM1.1',
        'bcc-csm1-1-m': 'BCC-CSM1.1(m)',
        'CESM1-BGC': 'CESM1(BGC)',
        'CESM1-CAM5': 'CESM1(CAM5)',
        'CESM1-CAM5-1-FV2': 'CESM1(CAM5.1,FV2)',
        'CESM1-FASTCHEM': 'CESM1(FASTCHEM)',
        'CESM1-WACCM': 'CESM1(WACCM)',
        'CSIRO-Mk3-6-0': 'CSIRO-Mk3.6.0',
        'fio-esm': 'FIO-ESM',
        'GFDL-CM2p1': 'GFDL-CM2.1',
        'inmcm4': 'INM-CM4',
        'MRI-AGCM3-2H': 'MRI-AGCM3.2H',
        'MRI-AGCM3-2S': 'MRI-AGCM3.2S'
    },
    'CMIP6': {},
    'CORDEX': {},
    'obs4MIPs': {},
}
"""Cache for the mapping between recipe/filesystem and ESGF dataset names."""


def create_dataset_map():
    """Create the DATASET_MAP from recipe datasets to ESGF dataset names.

    Run `python -m esmvalcore.esgf.facets` to print an up to date map.
    """
    cfg = get_esgf_config()
    search_args = dict(cfg["search_connection"])
    url = search_args.pop("urls")[0]
    connection = pyesgf.search.SearchConnection(url=url, **search_args)

    dataset_map = {}
    indices = {
        'CMIP3': 2,
        'CMIP5': 3,
        'CMIP6': 3,
        'CORDEX': 7,
        'obs4MIPs': 2,
    }

    for project in FACETS:
        dataset_map[project] = {}
        dataset_key = FACETS[project]['dataset']
        ctx = connection.new_context(
            project=project,
            facets=[dataset_key],
            fields=['id'],
            latest=True,
        )
        available_datasets = sorted(ctx.facet_counts[dataset_key])
        print(f"The following datasets are available for project {project}:")
        for dataset in available_datasets:
            print(dataset)

        # Figure out the ESGF name of the requested dataset
        n_available = len(available_datasets)
        for i, dataset in enumerate(available_datasets, 1):
            print(f"Looking for dataset name of facet name"
                  f" {dataset} ({i} of {n_available})")
            query = {dataset_key: dataset}
            dataset_result = next(iter(ctx.search(batch_size=1, **query)))
            print(f"Dataset id: {dataset_result.dataset_id}")
            dataset_id = dataset_result.dataset_id
            if dataset not in dataset_id:
                idx = indices[project]
                dataset_alias = dataset_id.split('.')[idx]
                print(f"Found dataset name '{dataset_alias}'"
                      f" for facet '{dataset}',")
                dataset_map[project][dataset_alias] = dataset

    return dataset_map


if __name__ == '__main__':
    # Run this module to create an up to date DATASET_MAP
    print(create_dataset_map())
