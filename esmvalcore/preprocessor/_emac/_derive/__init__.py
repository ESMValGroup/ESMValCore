"""Derivation module for EMAC variables."""

import importlib
import logging
from pathlib import Path

import iris

logger = logging.getLogger(__name__)

ALL_DERIVE_FUNCTIONS = {}


def get_derive_function(short_name):
    """Get derive function for EMAC CMORizer."""
    if not ALL_DERIVE_FUNCTIONS:
        for path in Path(__file__).parent.glob('[a-zA-Z0-9]*.py'):
            var = path.stem
            module = importlib.import_module(
                f'esmvalcore.preprocessor._emac._derive.{var}')
            if hasattr(module, 'derive'):
                ALL_DERIVE_FUNCTIONS[var] = getattr(module, 'derive')
            else:
                logger.warning(
                    "Derivation script %s.py for EMAC CMORization in "
                    "esmvalcore.preprocessor._emac._derive does not contain "
                    "necessary function 'derive(cubes)'", var)
    return ALL_DERIVE_FUNCTIONS.get(short_name)


def var_name_constraint(var_name):
    """:mod:`iris.Constraint` using `var_name` of an :mod:`iris.cube.Cube`."""
    return iris.Constraint(cube_func=lambda c: c.var_name == var_name)
