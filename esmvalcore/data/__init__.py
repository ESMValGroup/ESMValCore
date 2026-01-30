"""Find files using an intake-esm catalog and load them."""

from .intake._intake_dataset import clear_catalog_cache, load_catalogs
from .intake._interface import merge_intake_search_history

__all__ = [
    "clear_catalog_cache",
    "load_catalogs",
    "merge_intake_search_history",
]
