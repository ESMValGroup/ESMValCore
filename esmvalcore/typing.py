"""Type aliases for providing type hints."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from numbers import Number

import dask.array as da
import numpy as np
from iris.cube import Cube

FacetValue = str | Sequence[str] | Number | bool
"""Type describing a single facet."""

Facets = dict[str, FacetValue]
"""Type describing a collection of facets."""

NetCDFAttr = str | Number | Iterable
"""Type describing netCDF attributes.

`NetCDF attributes
<https://unidata.github.io/netcdf4-python/#attributes-in-a-netcdf-file>`_ can
be strings, numbers or sequences.
"""

DataType = np.ndarray | da.Array | Cube
"""Type describing data."""
