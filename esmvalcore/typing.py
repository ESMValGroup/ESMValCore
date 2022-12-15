"""Type aliases for providing type hints."""
from __future__ import annotations

from numbers import Number
from typing import Dict, Sequence, Union

FacetValue = Union[str, Sequence[str], Number]
"""Type describing a single facet."""

Facets = Dict[str, FacetValue]
"""Type describing a collection of facets."""
