"""Find files using an intake-esm catalog and load them."""

from ._dataset import IntakeDataset, load_catalogs

__all__ = ["IntakeDataset", "load_catalogs"]
