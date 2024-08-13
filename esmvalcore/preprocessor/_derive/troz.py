"""Derivation of variable ``troz``."""

from .toz import DerivedVariable as Toz
from .soz import DerivedVariable as Soz


class DerivedVariable(Toz):
    """Derivation of variable ``troz``."""

    @staticmethod
    def calculate(cubes):
        """Compute tropospheric column ozone.

        This is calculated as the difference between total column ozone (`toz`)
        and stratopheric column ozone (`soz`).

        """
        toz_cube = Toz.calculate(cubes)
        soz_cube = Soz.calculate(cubes)
        return toz_cube - soz_cube
