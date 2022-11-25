"""Types."""
from __future__ import annotations

from numbers import Number
from typing import Sequence, Union

FacetValue = Union[str, Sequence[str], Number]
Facets = dict[str, FacetValue]
