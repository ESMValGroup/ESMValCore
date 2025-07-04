"""Fixes for CESM2-WACCM model."""

from netCDF4 import Dataset

from esmvalcore.cmor._fixes.common import SiconcFixScalarCoord

from .cesm2 import Cl as BaseCl
from .cesm2 import Fgco2 as BaseFgco2
from .cesm2 import Omon as BaseOmon
from .cesm2 import Pr as BasePr
from .cesm2 import Tas as BaseTas
from .cesm2 import Tasmax as BaseTasmax
from .cesm2 import Tasmin as BaseTasmin


class Cl(BaseCl):
    """Fixes for cl."""

    def fix_file(self, file, output_dir, add_unique_suffix=False):
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
        new_path = self._fix_formula_terms(
            file,
            output_dir,
            add_unique_suffix=add_unique_suffix,
        )
        with Dataset(new_path, mode="a") as dataset:
            dataset.variables["a_bnds"][:] = dataset.variables["a_bnds"][
                :,
                ::-1,
            ]
            dataset.variables["b_bnds"][:] = dataset.variables["b_bnds"][
                :,
                ::-1,
            ]
        return new_path


Cli = Cl


Clw = Cl


Fgco2 = BaseFgco2


Omon = BaseOmon


Siconc = SiconcFixScalarCoord


Pr = BasePr


Tas = BaseTas


Tasmin = BaseTasmin


Tasmax = BaseTasmax
