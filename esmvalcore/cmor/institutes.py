from pathlib import Path

import yaml

from .table import CMOR_TABLES

with open(Path(__file__).parent / 'institutes.yml', 'r') as f:
    institutes = yaml.safe_load(f)


def get_institute(project, dataset):
    """See if the institute can be retrieved from the CMOR_TABLES, otherwise
    fall back to `institutes.yml` for CMIP3/CMIP5."""
    try:
        return CMOR_TABLES[project].institutes[dataset]
    except (KeyError, AttributeError):
        return institutes.get(project, {}).get(dataset, [])
