"""Type aliases for providing type hints."""
from __future__ import annotations

from numbers import Number
from typing import Dict, Iterable, Sequence, Union

FacetValue = Union[str, Sequence[str], Number]
"""Type describing a single facet."""

Facets = Dict[str, FacetValue]
"""Type describing a collection of facets."""

NetCDFAttrType = Union[str, Number, Iterable]
"""NetCDF attributes.

NetCDF attributes can be strings, numbers or sequences.
(https://unidata.github.io/netcdf4-python/#attributes-in-a-netcdf-file)
"""
