"""Find files using an intake-esm catalog and load them."""

from .intake._intake_dataset import clear_catalog_cache, load_catalogs
from .intake._interface import merge_intake_search_history

__all__ = [
    "load_catalogs",
    "clear_catalog_cache",
    "merge_intake_search_history",
]
