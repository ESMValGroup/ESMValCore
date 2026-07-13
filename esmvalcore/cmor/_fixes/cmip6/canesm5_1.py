"""Fixes for CanESM5-1 model."""

from esmvalcore.cmor._fixes.cmip6.canesm5 import Cl, Cli, Clw, Co2, Gpp, Ps


class Cl(Cl):
    """Fixes for cl."""


class Cli(Cli):
    """Fixes for cli."""


class Clw(Clw):
    """Fixes for clw."""


class Ps(Ps):
    """Fixes for ps."""


class Co2(Co2):
    """Fixes for co2."""


class Gpp(Gpp):
    """Fixes for gpp."""
