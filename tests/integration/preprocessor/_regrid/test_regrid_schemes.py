"""Integration tests for regrid schemes."""

import numpy as np
import pytest
from iris.cube import Cube

from esmvalcore.preprocessor.regrid_schemes import (
    GenericFuncScheme,
    GenericRegridder,
)


def set_data_to_const(cube, _, const=1.0):
    """Compute something to test ``GenericFuncScheme``."""
    return cube.copy(np.full(cube.shape, const))


@pytest.fixture
def generic_func_scheme():
    """Create a GenericFunctionScheme."""
    return GenericFuncScheme(set_data_to_const, const=2)


def test_generic_func_scheme_init(generic_func_scheme):
    """Test ``GenericFuncScheme``."""
    assert generic_func_scheme.func == set_data_to_const
    assert generic_func_scheme.kwargs == {"const": 2}


def test_generic_func_scheme_repr(generic_func_scheme):
    """Test ``GenericFuncScheme``."""
    repr_ = generic_func_scheme.__repr__()
    assert repr_ == "GenericFuncScheme(set_data_to_const, const=2)"


def test_generic_func_scheme_regridder(generic_func_scheme, mocker):
    """Test ``GenericFuncScheme``."""
    regridder = generic_func_scheme.regridder(
        mocker.sentinel.src_cube,
        mocker.sentinel.tgt_cube,
    )
    assert isinstance(regridder, GenericRegridder)
    assert regridder.src_cube == mocker.sentinel.src_cube
    assert regridder.tgt_cube == mocker.sentinel.tgt_cube
    assert regridder.func == set_data_to_const
    assert regridder.kwargs == {"const": 2}


def test_generic_func_scheme_regrid(generic_func_scheme, mocker):
    """Test ``GenericFuncScheme``."""
    cube = Cube([0.0, 0.0], var_name="x")

    result = cube.regrid(mocker.sentinel.tgt_grid, generic_func_scheme)

    assert result == Cube([2, 2], var_name="x")
