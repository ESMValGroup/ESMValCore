"""Fixes for CanESM5-1 model."""

from esmvalcore.cmor._fixes.cmip6.canesm5 import Cl as BaseCl
from esmvalcore.cmor._fixes.cmip6.canesm5 import Cli as BaseCli
from esmvalcore.cmor._fixes.cmip6.canesm5 import Clw as BaseClw
from esmvalcore.cmor._fixes.cmip6.canesm5 import Co2 as BaseCo2
from esmvalcore.cmor._fixes.cmip6.canesm5 import Gpp as BaseGpp
from esmvalcore.cmor._fixes.cmip6.canesm5 import Ps as BasePs


class Cl(BaseCl):
    """Fixes for cl."""


class Cli(BaseCli):
    """Fixes for cli."""


class Clw(BaseClw):
    """Fixes for clw."""


class Ps(BasePs):
    """Fixes for ps."""


class Co2(BaseCo2):
    """Fixes for co2."""


class Gpp(BaseGpp):
    """Fixes for gpp."""
