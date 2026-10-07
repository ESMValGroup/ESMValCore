"""Type aliases for providing type hints."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import dask.array as da
import numpy as np
from iris.cube import Cube

type FacetValue = str | Sequence[str] | int | float
"""Type describing a single facet."""

type Facets = dict[str, FacetValue]
"""Type describing a collection of facets."""

type NetCDFAttr = str | int | float | Iterable
"""Type describing netCDF attributes.

`NetCDF attributes
<https://unidata.github.io/netcdf4-python/#attributes-in-a-netcdf-file>`_ can
be strings, numbers or sequences.
"""

type DataType = np.ndarray | da.Array | Cube
"""Type describing data."""
