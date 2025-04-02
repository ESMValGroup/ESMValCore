"""Find files using an intake-esm catalog and load them."""

from .intake._intake_dataset import IntakeDataset, load_catalogs

__all__ = ["IntakeDataset", "load_catalogs"]
