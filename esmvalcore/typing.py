"""Type aliases for providing type hints."""
from __future__ import annotations

from numbers import Number
from typing import Dict, Iterable, Sequence, Union

import dask.array as da
import numpy as np
from iris.cube import Cube

FacetValue = Union[str, Sequence[str], Number]
"""Type describing a single facet."""

Facets = Dict[str, FacetValue]
"""Type describing a collection of facets."""

NetCDFAttr = Union[str, Number, Iterable]
"""Type describing netCDF attributes.

`NetCDF attributes
<https://unidata.github.io/netcdf4-python/#attributes-in-a-netcdf-file>`_ can
be strings, numbers or sequences.
"""

DataType = Union[np.ndarray, da.Array, Cube]
"""Type describing data."""
