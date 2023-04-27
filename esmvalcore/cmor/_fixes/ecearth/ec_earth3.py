"""Fixes for EC-Earth3 model."""
import cf_units
import numpy as np

from ..common import NemoGridFix
from ..fix import Fix
from ..shared import round_coordinates

Omon = NemoGridFix
