"""Fixes for BCC-ESM1 model."""
from ..cmip5.bcc_csm1_1 import Cl as BaseCl
from .bcc_csm2_mr import Tos as BaseTos


class Cl(BaseCl):
    """Fixes for ``cl``."""


class Clw(Cl):
    """Fixes for ``clw (same as for cl)``."""


class Cli(Cl):
    """Fixes for ``cli (same as for cl)``."""


class Tos(BaseTos):
    """Fixes for tos."""
