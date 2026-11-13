"""Fixes for CESM2-WACCM model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from esmvalcore.cmor._fixes.common import SiconcFixScalarCoord

from .cesm2 import Cl as BaseCl
from .cesm2 import Fgco2 as BaseFgco2
from .cesm2 import Omon as BaseOmon
from .cesm2 import Pr as BasePr
from .cesm2 import Tas as BaseTas
from .cesm2 import Tasmax as BaseTasmax
from .cesm2 import Tasmin as BaseTasmin

if TYPE_CHECKING:
    import ncdata


class Cl(BaseCl):
    """Fixes for cl."""

    @staticmethod
    def _fix_bounds_order(dataset: ncdata.NcData) -> None:
        """Correct the order of the bounds of ``a`` and ``b``."""
        for var_name in ("a_bnds", "b_bnds"):
            bounds = dataset.variables[var_name]
            bounds.data = bounds.data[:, ::-1]


Cli = Cl


Clw = Cl


Fgco2 = BaseFgco2


Omon = BaseOmon


Siconc = SiconcFixScalarCoord


Pr = BasePr


Tas = BaseTas


Tasmin = BaseTasmin


Tasmax = BaseTasmax
