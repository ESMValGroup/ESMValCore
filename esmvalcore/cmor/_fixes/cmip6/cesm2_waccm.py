"""Fixes for CESM2-WACCM model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import ncdata.netcdf4

from esmvalcore.cmor._fixes.common import SiconcFixScalarCoord
from esmvalcore.iris_helpers import dataset_to_iris

from .cesm2 import Cl as BaseCl
from .cesm2 import Fgco2 as BaseFgco2
from .cesm2 import Omon as BaseOmon
from .cesm2 import Pr as BasePr
from .cesm2 import Tas as BaseTas
from .cesm2 import Tasmax as BaseTasmax
from .cesm2 import Tasmin as BaseTasmin

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from iris.cube import Cube


class Cl(BaseCl):
    """Fixes for cl."""

    def fix_file(
        self,
        file: Path,
        output_dir: Path,  # noqa: ARG002
        add_unique_suffix: bool = False,  # noqa: ARG002
    ) -> Path | Sequence[Cube]:
        """Fix hybrid pressure coordinate.

        Adds missing ``formula_terms`` attribute to file.

        Note
        ----
        Fixing this with :mod:`iris` in ``fix_metadata`` or ``fix_data`` is
        **not** possible, since the bounds of the vertical coordinates ``a``
        and ``b`` are not present in the loaded :class:`iris.cube.CubeList`,
        even when :func:`iris.load_raw` is used.

        Parameters
        ----------
        file: str
            Path to the original file.
        output_dir: Path
            Output directory for fixed files.
        add_unique_suffix: bool, optional (default: False)
            Adds a unique suffix to `output_dir` for thread safety.

        Returns
        -------
        str
            Path to the fixed file.

        """
        dataset = ncdata.netcdf4.from_nc4(
            file,
            # Use iris-style chunks to avoid mismatching chunks between data
            # and derived coordinates, as the latter are automatically rechunked
            # by iris.
            dim_chunks={
                "time": "auto",
                "lev": None,
                "lat": None,
                "lon": None,
                "nbnd": None,
            },
        )
        self._fix_formula_terms(dataset)

        # Correct order of bounds data
        a_bnds = dataset.variables["a_bnds"]
        a_bnds.data = a_bnds.data[:, ::-1]
        b_bnds = dataset.variables["b_bnds"]
        b_bnds.data = b_bnds.data[:, ::-1]
        return [self.get_cube_from_list(dataset_to_iris(dataset, file))]


Cli = Cl


Clw = Cl


Fgco2 = BaseFgco2


Omon = BaseOmon


Siconc = SiconcFixScalarCoord


Pr = BasePr


Tas = BaseTas


Tasmin = BaseTasmin


Tasmax = BaseTasmax
