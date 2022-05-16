"""Preprocessor functions that do not fit into any of the categories."""

import logging
from collections import defaultdict

import dask.array as da

logger = logging.getLogger(__name__)


def clip(cube, minimum=None, maximum=None):
    """Clip values at a specified minimum and/or maximum value.

    Values lower than minimum are set to minimum and values
    higher than maximum are set to maximum.

    Parameters
    ----------
    cube: iris.cube.Cube
        iris cube to be clipped
    minimum: float
        lower threshold to be applied on input cube data.
    maximum: float
        upper threshold to be applied on input cube data.

    Returns
    -------
    iris.cube.Cube
        clipped cube.
    """
    if minimum is None and maximum is None:
        raise ValueError("Either minimum, maximum or both have to be\
                          specified.")
    elif minimum is not None and maximum is not None:
        if maximum < minimum:
            raise ValueError("Maximum should be equal or larger than minimum.")
    cube.data = da.clip(cube.core_data(), minimum, maximum)
    return cube


def _groupby(iterable, keyfunc):
    """Group iterable by key function.

    The items are grouped by the value that is returned by the `keyfunc`

    Parameters
    ----------
    iterable : list, tuple or iterable
        List of items to group
    keyfunc : callable
        Used to determine the group of each item. These become the keys
        of the returned dictionary

    Returns
    -------
    dict
        Returns a dictionary with the grouped values.
    """
    grouped = defaultdict(set)
    for item in iterable:
        key = keyfunc(item)
        grouped[key].add(item)

    return grouped


def _group_products(products, by_key):
    """Group products by the given list of attributes."""
    def grouper(product):
        return product.group(by_key)

    grouped = _groupby(products, keyfunc=grouper)
    return grouped.items()
