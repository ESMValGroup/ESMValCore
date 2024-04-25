# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test function :func:`iris.util.broadcast_to_shape`."""

from unittest import mock

import dask
import dask.array as da
import numpy as np
import numpy.ma as ma

from esmvalcore.preprocessor._regrid import broadcast_to_shape
from tests import assert_array_equal


def test_same_shape():
    # broadcast to current shape should result in no change
    a = np.random.random([2, 3])
    b = broadcast_to_shape(a, a.shape, (0, 1))
    assert_array_equal(b, a)


def test_added_dimensions():
    # adding two dimensions, on at the front and one in the middle of
    # the existing dimensions
    a = np.random.random([2, 3])
    b = broadcast_to_shape(a, (5, 2, 4, 3), (1, 3))
    for i in range(5):
        for j in range(4):
            assert_array_equal(b[i, :, j, :], a)


def test_added_dimensions_transpose():
    # adding dimensions and having the dimensions of the input
    # transposed
    a = np.random.random([2, 3])
    b = broadcast_to_shape(a, (5, 3, 4, 2), (3, 1))
    for i in range(5):
        for j in range(4):
            assert_array_equal(b[i, :, j, :].T, a)


@mock.patch.object(dask.base, "compute", wraps=dask.base.compute)
def test_lazy_added_dimensions_transpose(mocked_compute):
    # adding dimensions and having the dimensions of the input
    # transposed
    a = da.random.random([2, 3])
    b = broadcast_to_shape(a, (5, 3, 4, 2), (3, 1))
    mocked_compute.assert_not_called()
    for i in range(5):
        for j in range(4):
            assert_array_equal(b[i, :, j, :].T.compute(), a.compute())


def test_masked():
    # masked arrays are also accepted
    a = np.random.random([2, 3])
    m = ma.array(a, mask=[[0, 1, 0], [0, 1, 1]])
    b = broadcast_to_shape(m, (5, 3, 4, 2), (3, 1))
    for i in range(5):
        for j in range(4):
            assert_array_equal(b[i, :, j, :].T, m)


@mock.patch.object(dask.base, "compute", wraps=dask.base.compute)
def test_lazy_masked(mocked_compute):
    # masked arrays are also accepted
    a = np.random.random([2, 3])
    m = da.ma.masked_array(a, mask=[[0, 1, 0], [0, 1, 1]])
    b = broadcast_to_shape(m, (5, 3, 4, 2), (3, 1))
    mocked_compute.assert_not_called()
    for i in range(5):
        for j in range(4):
            assert_array_equal(b[i, :, j, :].compute().T, m.compute())


@mock.patch.object(dask.base, "compute", wraps=dask.base.compute)
def test_lazy_chunks(mocked_compute):
    # chunks can be specified along with the target shape and are only used
    # along new dimensions or on dimensions that have size 1 in the source
    # array.
    m = da.ma.masked_array(
        data=[[1, 2, 3, 4, 5]],
        mask=[[0, 1, 0, 0, 0]],
    ).rechunk((1, 2))
    b = broadcast_to_shape(
        m,
        dim_map=(1, 2),
        shape=(3, 4, 5),
        chunks=(
            1,  # used because target is new dim
            2,  # used because input size 1
            3,  # not used because broadcast does not rechunk
        ),
    )
    mocked_compute.assert_not_called()
    for i in range(3):
        for j in range(4):
            assert_array_equal(b[i, j, :].compute(), m[0].compute())
    assert b.chunks == ((1, 1, 1), (2, 2), (2, 2, 1))


def test_masked_degenerate():
    # masked arrays can have degenerate masks too
    a = np.random.random([2, 3])
    m = ma.array(a)
    b = broadcast_to_shape(m, (5, 3, 4, 2), (3, 1))
    for i in range(5):
        for j in range(4):
            assert_array_equal(b[i, :, j, :].T, m)
