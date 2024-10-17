"""Fixes for CESM2-WACCM model."""

import ncdata.iris
import ncdata.netcdf4

from ..common import SiconcFixScalarCoord
from .cesm2 import Cl as BaseCl
from .cesm2 import Fgco2 as BaseFgco2
from .cesm2 import Omon as BaseOmon
from .cesm2 import Tas as BaseTas


class Cl(BaseCl):
    """Fixes for ``cl``."""

    def fix_file(self, filepath, output_dir, add_unique_suffix=False):
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
        filepath : str
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
        dataset = ncdata.netcdf4.from_nc4(filepath)
        a_bnds = dataset.variables["a_bnds"]
        a_bnds.data = a_bnds.data[:, ::-1]
        b_bnds = dataset.variables["b_bnds"]
        b_bnds.data = b_bnds.data[:, ::-1]
        lev = dataset.variables["lev"]
        lev.set_attrval("formula_terms", "p0: p0 a: a b: b ps: ps")
        lev.set_attrval(
            "standard_name", "atmosphere_hybrid_sigma_pressure_coordinate"
        )
        cubes = ncdata.iris.to_iris(dataset)
        return cubes


Cli = Cl

Clw = Cl

Fgco2 = BaseFgco2

Omon = BaseOmon

Siconc = SiconcFixScalarCoord

Tas = BaseTas
